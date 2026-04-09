# neet-crypto-trader

Budget-controlled crypto trading agent for OKX. Automates short-term speculative trading with strict weekly/monthly budget limits on losses and gains.

## Features

- **OKX Integration** — REST + WebSocket via official `python-okx` SDK
- **Budget Controls** — Weekly/monthly capital limits with automatic stop when thresholds hit
- **Risk Management** — Triple barrier on every position (stop-loss + take-profit + time limit), server-side stop-losses that survive bot crashes
- **Paper Trading** — Demo mode by default using OKX's sandbox environment
- **Pluggable Strategies** — Abstract strategy interface; ship with RSI + MACD momentum strategy
- **Persistent State** — SQLite-backed budget tracking survives restarts

## Quick Start

### Prerequisites

- Python 3.11+
- OKX account with API key ([create one here](https://www.okx.com/account/my-api))
  - Enable **Trade** permission only
  - **Never** enable Withdraw permission

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
```
OKX_API_KEY=your-key
OKX_API_SECRET=your-secret
OKX_PASSPHRASE=your-passphrase
OKX_DEMO_MODE=true
```

**`config/default.toml`** — Trading parameters:
```toml
[trading]
pairs = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
strategy = "momentum"
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
```

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
  risk/
    budget_manager.py # Weekly/monthly budget tracking
    position_sizer.py # Per-trade sizing
    risk_manager.py   # Central risk gate
    protections.py    # Circuit breakers (StoplossGuard, MaxDrawdown)
  portfolio/
    tracker.py        # Position tracking, P&L
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/

# Run the bot (demo mode)
nct
```

## Safety

- Paper trading is the **default** — live trading requires explicitly setting `OKX_DEMO_MODE=false`
- Every position has a **server-side stop-loss** on OKX (executes even if bot is offline)
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
- **Phase 6** — Hardening (backtesting, Telegram bot, Docker, second strategy, LLM layer)

## License

Private project.
