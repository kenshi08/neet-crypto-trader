# neet-crypto-trader

Budget-controlled crypto trading agent supporting **Coinbase**, **OKX**, and **Bybit**. Automates short-term speculative trading with strict weekly/monthly budget limits on losses and gains.

## Features

- **Multi-exchange** — Switch between Coinbase, OKX, and Bybit via the `EXCHANGE` env var
- **Budget Controls** — Weekly/monthly capital limits with automatic stop when thresholds hit
- **Risk Management** — Triple barrier on every position (stop-loss + take-profit + time limit), atomic entry reversal if SL placement fails, server-side stop-losses that survive bot crashes
- **Paper Trading** — Demo/testnet mode by default, with unambiguous `PAPER_DRY_RUN` / `DEMO_REAL_BALANCE` / `LIVE_REAL_MONEY` mode announcement at startup
- **Realistic Backtests** — Fee-aware P&L with per-side fee configuration and exchange presets
- **Pluggable Strategies** — Config-driven selection between RSI+MACD momentum and Bollinger Bands mean reversion
- **Persistent State** — SQLite-backed budget tracking survives restarts
- **Telegram Bot** — Real-time trade notifications, status commands, kill switch

## Quick Start

### Prerequisites

- Python 3.11+
- One of:
  - **Bybit** account ([live](https://www.bybit.com) or [testnet](https://testnet.bybit.com)) — recommended
  - **OKX** account ([create here](https://www.okx.com/account/my-api))
- API key with **Trade** permission only — **never** enable Withdraw

### Installation

```bash
git clone https://github.com/kenshi08/neet-crypto-trader.git
cd neet-crypto-trader
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

1. Copy the environment template and add your OKX API credentials:
```bash
cp .env.example .env
# Edit .env with your API key, secret, and passphrase
```

2. Review trading config in `config/default.toml` — adjust pairs, budget, and risk parameters.

3. Run in demo mode (default — uses fake funds):
```bash
nct
```

### Configuration Reference

**`.env`** — Credentials (never committed to git):

For Coinbase (recommended for Singapore):
```
EXCHANGE=coinbase
COINBASE_API_KEY=organizations/xxx/apiKeys/yyy
COINBASE_API_SECRET="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----\n"
COINBASE_DEMO_MODE=true
```

For Bybit (geo-blocked in some regions including Singapore):
```
EXCHANGE=bybit
BYBIT_API_KEY=your-key
BYBIT_API_SECRET=your-secret
BYBIT_DEMO_MODE=true
```

For OKX:
```
EXCHANGE=okx
OKX_API_KEY=your-key
OKX_API_SECRET=your-secret
OKX_PASSPHRASE=your-passphrase
OKX_DEMO_MODE=true
# Set OKX_BASE_URL=https://my.okx.com if you registered in Europe (EEA)
```

**`config/default.toml`** — Trading parameters:
```toml
[trading]
pairs = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
strategy = "momentum"            # or "mean_reversion"
timeframe = "15m"
max_open_positions = 3

[budget]
period = "weekly"
amount_usdt = 500
max_loss_pct = 5.0
max_gain_pct = 15.0

[risk]
stop_loss_pct = 3.0
take_profit_pct = 5.0
time_limit_seconds = 3600

# Strategy-specific parameters under [strategy.<name>]
# The bot reads the section matching `trading.strategy` above.

[strategy.momentum]
rsi_period = 14
rsi_oversold = 30.0
rsi_overbought = 70.0
macd_fast = 12
macd_slow = 26
macd_signal = 9

[strategy.mean_reversion]
bb_period = 20
bb_std = 2.0
volume_multiplier = 1.2
```

### Switching Strategies

Change `strategy = "momentum"` to `strategy = "mean_reversion"` in your TOML
config and restart the bot. The strategy factory picks the right implementation
at startup. Available strategies: `momentum`, `mean_reversion`.

## Project Structure

```
src/nct/
  config.py           # Pydantic config (TOML + .env)
  exceptions.py       # Exception hierarchy
  db.py               # SQLite persistence
  main.py             # Entry point / orchestrator
  exchange/
    client.py         # OKX API wrapper (@retrier, rate limiting, dry-run)
    models.py         # Typed data models (Ticker, Candle, Order, Position)
    market_feed.py    # WebSocket real-time data
  strategy/
    base.py           # IStrategy ABC
    momentum.py       # RSI + MACD strategy
    mean_reversion.py # Bollinger Bands + volume strategy
    data_provider.py  # OHLCV fetching, caching, DataFrame conversion
  risk/
    budget_manager.py # Weekly/monthly budget tracking
    position_sizer.py # Per-trade sizing
    risk_manager.py   # Central risk gate
    protections.py    # Circuit breakers (StoplossGuard, MaxDrawdown)
  portfolio/
    tracker.py        # Position tracking, P&L
  executor.py         # Order execution with triple barrier
  notifier.py         # Telegram bot (optional)
  logging_setup.py    # Structured logging config
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/

# Run the bot (demo mode)
nct

# Run backtester (fee-aware — defaults to Coinbase 0.4% per side)
python scripts/backtest.py --pair BTC-USDT --timeframe 15m --limit 300

# Override fees for a different exchange
python scripts/backtest.py --pair BTC-USDT --exchange okx       # 0.1% per side
python scripts/backtest.py --pair BTC-USDT --fee-pct 0.05       # custom rate

# Deploy with Docker
docker compose up -d
```

## Safety

- Paper trading is the **default** — live trading requires explicitly setting `<EXCHANGE>_DEMO_MODE=false`
- Startup emits an unmistakable `running_mode` log line: `PAPER_DRY_RUN`, `DEMO_REAL_BALANCE`, or `LIVE_REAL_MONEY` (and a matching Telegram header)
- Every position has a **server-side stop-loss** (executes even if bot is offline). If stop-loss placement fails, the entry is automatically reversed — no unprotected positions.
- Budget limits are **hard-enforced** — no code path can bypass the BudgetManager
- API keys never appear in logs or config files
- Graceful shutdown on SIGINT/SIGTERM cancels open orders and persists state

## Deployment

Runs on any always-on machine: Synology NAS, VPS ($4/mo Hetzner), or local.

```bash
docker compose up -d
```

See issue tracker for detailed deployment docs (#20).

## Roadmap

See [GitHub Issues](https://github.com/kenshi08/neet-crypto-trader/issues) for the full roadmap, organized by phase:

- **Phase 1** — Foundation (config, OKX client, data models) — **Done**
- **Phase 2** — Risk Engine (budget manager, position sizer, protections)
- **Phase 3** — Strategy Engine (IStrategy ABC, momentum strategy)
- **Phase 4** — Execution (order executor, portfolio tracker, WebSocket feed)
- **Phase 5** — Main Loop (orchestrator, graceful shutdown, structured logging)
- **Phase 6** — Hardening (backtesting, Telegram bot, Docker, second strategy, LLM layer) — **Done**

## License

Private project.
