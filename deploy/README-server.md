# AlphaMaster 服务器部署 runbook（香港 Debian 12 / paper autopilot）

目标：在一台已有的 Debian 12 / 2C2G / 香港服务器上，长时间运行 paper 模式 autopilot，
通过 web 看板（Tailscale 私网访问）观察策略效果。autopilot 子进程由 web 的 β 自动续命守护。

> 前置：本分支 `feat/autopilot-server-deploy` 已含 β 自动续命代码 + 本部署产物。

## 0. 前提

- 一台 Debian 12 服务器（2C2G，香港），有 root/sudo。
- 本分支已 push 到一个服务器能 clone 的 git 远端（自建 / GitHub / Gitee）。下文 `<repo>` 替换为它。
- 你的笔记本/手机用于看板，会加入同一 Tailscale 私网。

## 1. 系统依赖 + 用户（root）

```bash
apt update && apt install -y python3.11-venv python3-dev build-essential git curl
# Debian 12 自带 Python 3.11；确认：
python3.11 --version

# 专用非 root 用户
useradd -m -s /bin/bash alphamaster
mkdir -p /srv/alphamaster /srv/alphamaster/kline_cache
chown -R alphamaster:alphamaster /srv/alphamaster

# NTP：bar 时间戳 / "剔未收盘 bar" 依赖准点时钟
timedatectl set-ntp true
timedatectl status
```

## 2. 拉代码 + 装依赖（切到 alphamaster）

```bash
sudo -iu alphamaster
cd /srv/alphamaster
git clone -b feat/autopilot-server-deploy <repo> /srv/alphamaster

python3.11 -m venv venv
venv/bin/pip install --upgrade pip

# torch 单独强制 CPU wheel（关键：否则 Linux 默认拉 CUDA ~2GB，2G 机器装时就 OOM）
venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch

# 精简依赖（不含 MetaTrader5 / tvdatafeed / matplotlib / playwright 等重且非必需包）
venv/bin/pip install -r deploy/requirements-server.txt
```

## 3. 导入冒烟（验证依赖齐全，web 能起）

```bash
venv/bin/python -c "import web.app; print('import OK')"
```

若报某个被排除的包缺失（探查表明不应发生），把它补进 `deploy/requirements-server.txt` 重装。

## 4. 配置 .env（paper 无需密钥）

```bash
cp .env.example .env
# paper 模式不需要 OKX 凭据，留空即可；只用 OKX 公开行情 REST。
```

## 5. systemd 服务

```bash
sudo cp deploy/alphamaster-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now alphamaster-web
sudo systemctl status alphamaster-web --no-pager
curl -s http://127.0.0.1:8765/api/health
```

应看到 `active (running)` 且 `/api/health` 返回 ok。

## 6. Tailscale（私网访问零鉴权看板）

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up                 # 浏览器授权本机入网
sudo tailscale serve --bg --https=443 http://127.0.0.1:8765
tailscale status                  # 记下本机 tailnet 名，如 amhk.xxx.ts.net
```

- web 仍绑 `127.0.0.1`（零公网端口），`tailscale serve` 把它代理到 tailnet 的 HTTPS。
- 在你的笔记本/手机装 Tailscale、加入同一私网，开 `https://amhk.xxx.ts.net/` 即看板。
- 备选访问（不依赖 tailscale serve 的 HTTPS 证书）：`ssh -L 8765:127.0.0.1:8765 alphamaster@<tailnet-ip>` 后开 `http://localhost:8765`。

## 7. 开跑 + 观察

看板 → **自动驾驶** → 选一个币（BTC/ETH/SOL/XRP/DOGE 之一）→ mode=paper → 启动。
- 下一根 H1 bar 收盘后，`autopilot_state.json` 出现、`autopilot_child.pid` 出现、看板显示目标/实际仓位。
- `journalctl -u alphamaster-web -f` 看 β 续命/重启决策日志；`logs/autopilot_*.log` 看引擎逐 bar 日志。

## 8. 日常运维

- **更新代码**：`cd /srv/alphamaster && git pull && sudo systemctl restart alphamaster-web`。
  web 重启 → β 见 `autopilot_intended_running=True` → 自动重拉 autopilot。
- **回撤熔断**：已靠 systemd env `AUTOPILOT_BREAKER_MAX_DRAWDOWN_PCT=-2.0` 关掉，曲线完整跑。
  连通性熔断（OKX 断 3 次）仍生效 → halt → β 退避重拉 → 恢复后自愈。
- **别在 UI 随手切币**：换标的 = `autopilot_state.json` 丢弃重建，几周观察数据清零。盯一个到底。
- **6 周提醒**：`autopilot_state.json` 的 history/trades cap 1000（H1≈42 天 FIFO）。
  想留早期数据，第 6 周前补追加式 CSV（本部署暂未含；见 plan Q7=d）。

## 9. 验证 β（部署后抽查）

- `sudo systemctl restart alphamaster-web`：子进程被 cgroup-kill → β 开机重拉（flag=True 时）；`journalctl` 见 `boot relaunch: 已重拉 ...`。
- 手动 `kill -9 <autopilot 子进程 pid>`（看 `autopilot_child.pid`）：watcher 在 ~10s 内重拉，日志见 `watcher: 10s 后重拉`。
- （可选）服务器 reboot → systemd 自启 web → β 自启 autopilot。
