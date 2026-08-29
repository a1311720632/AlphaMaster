"""Alerter 节流 + engine 心跳/摘要测试（B4/B5/ADR-0007）。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from autopilot.alerts import Alerter


class _Recorder:
    def __init__(self):
        self.sent: list[str] = []

    def __call__(self, text: str):
        self.sent.append(text)
        return (True, "ok")


def _alerter(monkeypatch, recorder=None) -> tuple[Alerter, _Recorder]:
    rec = recorder or _Recorder()
    monkeypatch.setattr("web.feishu_notify.send_text", lambda text, **kw: rec(text))
    # 测试默认"已配置"（settings 被 conftest 隔离成空 → configured 读真实文件会为 False）
    monkeypatch.setattr(
        Alerter, "configured", property(lambda self: True)
    )
    return Alerter(log=lambda m: None), rec


def test_disabled_config_short_circuits(monkeypatch):
    """开关关/无 webhook → send_text 不被调用（事件仅入账本，log 一条）。"""
    al = Alerter(log=lambda m: None)
    monkeypatch.setattr(Alerter, "configured", property(lambda self: False))
    sent: list[str] = []
    monkeypatch.setattr("web.feishu_notify.send_text", lambda text, **kw: (sent.append(text), (True, "ok"))[1])
    al.send("k", "x", critical=True)  # 即使 critical 也不发
    assert sent == []


def test_throttle_same_key_sends_once(monkeypatch):
    """同 key 重复触发只发一次（节流）；恢复后再触发才再发。"""
    al, rec = _alerter(monkeypatch)
    al.send("exec_fail", "第一次")
    al.send("exec_fail", "第二次（应被节流）")
    assert len(rec.sent) == 1
    al.resolve("exec_fail", "已恢复")
    assert len(rec.sent) == 2
    al.send("exec_fail", "恢复后再触发")
    assert len(rec.sent) == 3


def test_critical_bypasses_throttle(monkeypatch):
    """critical=True 无视节流必发（熔断类天然低频，宁重复不漏发）。"""
    al, rec = _alerter(monkeypatch)
    al.send("breaker", "熔断A", critical=True)
    al.send("breaker", "熔断B", critical=True)
    assert len(rec.sent) == 2


def test_different_keys_independent(monkeypatch):
    al, rec = _alerter(monkeypatch)
    al.send("key1", "a")
    al.send("key2", "b")
    assert len(rec.sent) == 2


def test_send_failure_silent(monkeypatch):
    """发送失败不抛（告警失败不阻断 halt/交易路径）。"""
    monkeypatch.setattr(
        "web.feishu_notify.send_text", lambda text, **kw: (_ for _ in ()).throw(RuntimeError("net"))
    )
    al = Alerter(log=lambda m: None)
    al.send("k", "x", critical=True)  # 不应抛
    al.resolve("k")  # 同样不应抛


def test_heartbeat_ping(monkeypatch, tmp_path):
    """心跳：URL 命中、5s 超时、成功后 max_silent 窗口内不重发。"""
    from web.data_sources.base import Bar, DataSource

    from autopilot.backends import ExecutionBackend, OrderResult
    from autopilot.engine import AutopilotEngine
    from autopilot.strategy_loader import StrategySpec

    calls: list[str] = []

    class FakeSource(DataSource):
        kind = "fake"

        def available(self):
            return (True, "fake")

        def supported_timeframes(self):
            return ["1h"]

        def preset_symbols(self):
            return ["T"]

        def fetch_bars(self, symbol, timeframe, n, drop_forming=True):
            now = 1_700_000_000
            return [
                Bar(ts=now - (n - 1 - i) * 3600, open=100, high=101, low=99, close=100, volume=1)
                for i in range(n)
            ]

    class NoTradeBackend(ExecutionBackend):
        mode = "paper"

        def fetch_equity(self):
            return 100.0

        def fetch_position_notional(self, symbol):
            return 0.0

        def place_delta_order(self, symbol, delta_notional):
            return OrderResult(ok=True, filled_notional=delta_notional)

        def flatten_all(self, symbol):
            return OrderResult(ok=True)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        assert timeout == 5
        return _Resp()

    monkeypatch.setattr("autopilot.engine.urllib.request.urlopen", fake_urlopen)

    eng = AutopilotEngine(
        strategy=StrategySpec(formula=[0], symbol="T", timeframe="1h", score=1.0,
                              vocab_version="t", path=""),
        datasource=FakeSource(),
        backend=NoTradeBackend(),
        lookback_bars=250,
        breaker_max_drawdown_pct=-0.5,
        breaker_max_bars_stale=99,
        min_notional_delta=1e18,  # 永不触发下单
        state_path=tmp_path / "s.json",
        stop_signal_paths=[tmp_path / "stop"],
        heartbeat_url="https://hc.example.com/ping/test",
        heartbeat_max_silent_s=900,
        log=lambda m: None,
    )
    eng._maybe_heartbeat()
    eng._maybe_heartbeat()  # 窗口内 → 不重发
    assert calls == ["https://hc.example.com/ping/test"]
    # 喂一个"很久以前"的成功时间 → 再 ping 必发
    eng._hb_last_ok -= 1000
    eng._maybe_heartbeat()
    assert len(calls) == 2


# ── 每日日报（壁钟驱动，UTC 日切 +5min）─────────────────────────────

_D0, _D1, _D2 = "2026-08-26", "2026-08-27", "2026-08-28"


def _utc(day: str, hh: int = 0, mm: int = 0, ss: int = 0) -> int:
    y, m, d = (int(x) for x in day.split("-"))
    return int(datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc).timestamp())


class _StaticBackend:
    """固定持仓的 paper backend（供日报取仓位明细）。"""

    mode = "paper"

    def __init__(self, position=(0.0, 0.0, 0.0)):
        self._pos = position

    def fetch_equity(self):
        return 100.0

    def fetch_position_notional(self, symbol):
        return self._pos[0]

    def fetch_position_detail(self, symbol):
        return self._pos

    def place_delta_order(self, symbol, delta_notional):
        from autopilot.backends import OrderResult

        return OrderResult(ok=True, filled_notional=delta_notional)

    def flatten_all(self, symbol):
        from autopilot.backends import OrderResult

        return OrderResult(ok=True)


def _fill_day(eng, day: str, equities: list[float]) -> None:
    """往 hot state 灌某 UTC 日的逐时权益（bar ts = 开盘时间）。"""
    for i, eq in enumerate(equities):
        eng.state.history.append({
            "ts": _utc(day, i),
            "equity": eq,
            "actual_notional": 500.0,
            "entry_price": 100.0,
            "unrealized_pnl": 2.5,
        })


def _digest_engine(monkeypatch, tmp_path, backend=None):
    from web.data_sources.base import Bar, DataSource

    from autopilot.engine import AutopilotEngine
    from autopilot.strategy_loader import StrategySpec

    class FakeSource(DataSource):
        kind = "fake"

        def available(self):
            return (True, "fake")

        def supported_timeframes(self):
            return ["1h"]

        def preset_symbols(self):
            return ["T"]

        def fetch_bars(self, symbol, timeframe, n, drop_forming=True):
            now = 1_700_000_000
            return [
                Bar(ts=now - (n - 1 - i) * 3600, open=100, high=101, low=99, close=100, volume=1)
                for i in range(n)
            ]

    al, rec = _alerter(monkeypatch)
    eng = AutopilotEngine(
        strategy=StrategySpec(formula=[0], symbol="T", timeframe="1h", score=1.0,
                              vocab_version="t", path=""),
        datasource=FakeSource(),
        backend=backend if backend is not None else _StaticBackend((500.0, 100.0, 2.5)),
        lookback_bars=250,
        breaker_max_drawdown_pct=-0.5,
        breaker_max_bars_stale=99,
        min_notional_delta=1e18,
        state_path=tmp_path / "s.json",
        stop_signal_paths=[tmp_path / "stop"],
        log=lambda m: None,
        alerter=al,
    )
    return eng, rec


def test_digest_sends_every_day_not_once(monkeypatch, tmp_path):
    """回归：日报连发两天（旧实现同 key 节流后永远静默，只发第一份）。"""
    eng, rec = _digest_engine(monkeypatch, tmp_path)
    _fill_day(eng, _D0, [99.0, 100.0])
    _fill_day(eng, _D1, [100.0, 105.0])
    eng._maybe_daily_digest(now_s=_utc(_D1, 12))   # 首调仅记日，不发
    assert rec.sent == []
    eng._maybe_daily_digest(now_s=_utc(_D2, 0, 5))  # D2 日切 → D1 日报
    assert len(rec.sent) == 1
    eng._maybe_daily_digest(now_s=_utc(_D2, 12))   # 同日再调不重发
    assert len(rec.sent) == 1
    _fill_day(eng, _D2, [105.0, 110.0])
    eng._maybe_daily_digest(now_s=_utc("2026-08-29", 0, 5))  # D3 日切 → D2 日报
    assert len(rec.sent) == 2


def test_digest_5min_gate(monkeypatch, tmp_path):
    """日切后 5min 内不发；到点即发；且闸门不吞掉日切标记。"""
    eng, rec = _digest_engine(monkeypatch, tmp_path)
    _fill_day(eng, _D0, [99.0, 100.0])
    _fill_day(eng, _D1, [100.0, 105.0])
    eng._maybe_daily_digest(now_s=_utc(_D1, 12))    # 首调记日
    eng._maybe_daily_digest(now_s=_utc(_D2, 0, 4, 59))
    assert rec.sent == []
    eng._maybe_daily_digest(now_s=_utc(_D2, 0, 5))
    assert len(rec.sent) == 1


def test_digest_content(monkeypatch, tmp_path):
    """口径：昨收/前收 close-to-close、实时仓位、无回撤字段、[autopilot] 前缀保关键词过滤。"""
    eng, rec = _digest_engine(monkeypatch, tmp_path)
    _fill_day(eng, _D0, [99.0, 100.0])   # 前收 = 100.0
    _fill_day(eng, _D1, [100.0, 105.0])  # 昨收 = 105.0 → 当日 +5.00%
    eng.state.trades.append({"ts": _utc(_D1, 10)})
    eng.state.trades.append({"ts": _utc(_D1, 15)})
    eng._maybe_daily_digest(now_s=_utc(_D1, 12))
    eng._maybe_daily_digest(now_s=_utc(_D2, 0, 5))
    assert len(rec.sent) == 1
    text = rec.sent[0]
    assert text.startswith("[autopilot][T]")
    assert f"{_D1} 日报" in text
    assert "权益 105.00" in text
    assert "当日 +5.00%" in text          # (昨收105 - 前收100)/前收100
    assert "小赚" in text
    assert "回撤" not in text             # 不再展示回撤口径
    assert "多头 500.00 USDT" in text     # 实时仓位
    assert "开仓价 100.00" in text
    assert "成交 2 笔" in text


def test_digest_first_day_fallback_and_flat(monkeypatch, tmp_path):
    """上线首日无前日数据 → 退回昨日首根权益作基准；空仓文案正常。"""
    eng, rec = _digest_engine(monkeypatch, tmp_path, backend=_StaticBackend((0.0, 0.0, 0.0)))
    _fill_day(eng, _D1, [100.0, 101.0])  # 无 D0 → 基准退回 D1 首根 100 → +1.00%
    eng._maybe_daily_digest(now_s=_utc(_D1, 12))
    eng._maybe_daily_digest(now_s=_utc(_D2, 0, 5))
    assert len(rec.sent) == 1
    assert "当日 +1.00%" in rec.sent[0]
    assert "空仓" in rec.sent[0]


def test_digest_no_data_silent(monkeypatch, tmp_path):
    """昨日无 bar → 静默跳过（数据不足不发，事件不炸）。"""
    eng, rec = _digest_engine(monkeypatch, tmp_path)
    eng._maybe_daily_digest(now_s=_utc(_D1, 12))
    eng._maybe_daily_digest(now_s=_utc(_D2, 0, 5))
    assert rec.sent == []
