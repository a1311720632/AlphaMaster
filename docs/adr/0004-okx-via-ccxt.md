# 加密执行：OKX 起步、走 ccxt、后端交易所无关

自动驾驶 v1 第一个交易所选 **OKX**（已有只读 `OKXSource`、有成熟 demo trading 模拟盘可作 testnet），下单统一走 **ccxt**（`set_sandbox_mode(True)` 切 testnet，`create_order` / `fetch_positions` / `fetch_funding_rate` 统一接口）。执行后端接口保持**交易所无关**，加新交易所只是新增一个后端实现。

## 为什么

用户要求“主要连接加密货币交易所”（复数）。ccxt 的价值就是一套 API 适配多所；日后加 Binance/Bybit 不必重写执行层。OKX 作起点复用了已有的 `OKXSource` 市场数据集成，且其 demo trading 环境成熟，testnet 模式成本低。

## 代价 / 后果

- ccxt 统一 API 在 OKX perp 上有若干怪癖：合约命名（`BTC/USDT:USDT`）、持仓模式（单向 vs 双向 / hedge vs one-way）、保证金模式（逐仓 vs 全仓）需在后端实现里显式设定，不能裸用默认值。
- ccxt 成为新增依赖（requirements.txt），版本升级偶尔有 break change，需在 CI 锁版本。
- 与现有 `OKXSource`（市场数据，requests 自实现）并存：市场数据继续用 OKXSource，交易用 ccxt，两者不强行合并，避免改造现有数据通路。

## 放弃的方案

- OKX 官方 python SDK：最贴近 OKX 语义、抽象泄漏少，但锁死 OKX，换所要重写执行层，弃用 ccxt 的多所能力。
- Binance + ccxt：流动性最好、testnet 成熟，但放弃已有 OKXSource 集成，且 Binance 的地区合规限制需单独确认，第一步范围变大。
