"""备用源链（Fallback Chain，B2/ADR-0007）：OKX → Bybit → Binance。

行情断连是业务连续性问题（有替补）；交易断连是资金安全问题（无替补，走执行熔断）。

链语义（计数按**调用次**而非调用内尝试次——engine 的 cadence 轮询提供天然间隔）：
  - 每次 fetch 从 active 源起逐家试 1 次，任一家成功即供数并清失败计数。
  - active 源失败一次计 1；连续失败达 max_fails_per_source 次（跨调用）→ active
    前移到下一家并记切换事件。备源活跃期间主源恢复（供数源更靠前）→ 切回。
  - 平价切换：fetch_bars 返回值**单一来源**——备用源供全窗口 bar，绝不与主源历史
    混拼（混拼会让成交量类特征跨源跳变）。切换期间的信号漂移属执行摩擦（ADR-0003）。
  - 单次调用全链失败 → raise DataSourceUnavailable（engine 的 ConnectivityBreaker
    兜住——只有全挂才计断连熔断）。
  - 事件外抛拉取式：切换后 last_switch 置 (from_kind, to_kind, ts)，engine 每 tick
    消费（读后置 None）——避免回调侵入数据源接口。
"""
from __future__ import annotations

import time

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable


class FallbackDataSource(DataSource):
    kind = "fallback"
    label = "备用源链"

    def __init__(self, sources: list[DataSource], max_fails_per_source: int = 3) -> None:
        if not sources:
            raise ValueError("备用源链至少需要一个数据源")
        self._sources = list(sources)
        self._max_fails = max(1, int(max_fails_per_source))
        self._active_idx = 0        # 当前 active 源（链的起点）
        self._fails = 0             # active 源连续失败计数（跨调用，成功清零）
        self._served_idx: int | None = None  # 上次实际供数的源下标
        # 最近一次切换事件：(from_kind, to_kind, wall_ts)；engine 读后置 None
        self.last_switch: tuple[str, str, int] | None = None

    # ── 查询接口（聚合语义）────────────────────────────────────────────
    def available(self) -> tuple[bool, str]:
        ok, hint = self._sources[self._active_idx].available()
        chain = " → ".join(s.kind for s in self._sources)
        return (ok, f"{hint}（链: {chain}）")

    def supported_timeframes(self) -> list[str]:
        # 链内所有源都支持的周期（交集）——保证任何一家接手都能供数
        sets = [set(s.supported_timeframes()) for s in self._sources]
        common = set.intersection(*sets) if sets else set()
        return sorted(common, key=list(self._sources[0].supported_timeframes()).index)

    def preset_symbols(self) -> list[str]:
        return self._sources[0].preset_symbols()

    # ── 链切换核心 ─────────────────────────────────────────────────────
    def _sweep(self, method: str, *args):
        """从 active 起逐家试一圈。返回 (成功下标, 结果)；全链失败 (None, None)。

        active 源失败计 1（跨调用累计）；达阈值 → 切下一家 + 记事件。
        非活跃源（本调用内途经的备源）失败不计数——它们只是本轮没顶上。
        """
        start = self._active_idx
        for off in range(len(self._sources)):
            idx = (start + off) % len(self._sources)
            try:
                result = getattr(self._sources[idx], method)(*args)
            except Exception:  # noqa: BLE001 - 任一源失败都走链逻辑
                if idx == self._active_idx:
                    self._fails += 1
                    if self._fails >= self._max_fails:
                        nxt = (self._active_idx + 1) % len(self._sources)
                        self.last_switch = (
                            self._sources[self._active_idx].kind,
                            self._sources[nxt].kind,
                            int(time.time()),
                        )
                        self._active_idx = nxt
                        self._fails = 0
                continue
            # 成功：active 源自己成功才清它的失败计数（备源顶上不清——active 的病要自己好）；
            # 供数源比上次更靠前 → 切回（含备源→主源）
            if idx == self._active_idx:
                self._fails = 0
            prev = self._served_idx
            if prev is not None and idx < prev:
                self.last_switch = (
                    self._sources[prev].kind, self._sources[idx].kind, int(time.time())
                )
            self._served_idx = idx
            return idx, result
        return None, None

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        _, bars = self._sweep("fetch_bars", symbol, timeframe, n, drop_forming)
        if not bars:
            raise DataSourceUnavailable(
                f"备用源链全部失败: {' → '.join(s.kind for s in self._sources)}"
            )
        return bars

    def fetch_ticker(self, symbol: str) -> float:
        _, price = self._sweep("fetch_ticker", symbol)
        if not price or price <= 0:
            raise DataSourceUnavailable("备用源链 ticker 全部失败")
        return float(price)
