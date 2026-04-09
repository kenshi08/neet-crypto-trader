# CLAUDE.md — neet-crypto-trader

## Project Overview

A crypto trading agent for OKX that automates short-term speculative trading with strict budget controls (weekly/monthly limits on losses and gains). Designed for a single user managing a personal trading budget.

## Tech Stack

- **Python 3.12+**
- **python-okx** — Official OKX SDK (REST + WebSocket)
- **pandas + pandas-ta** — OHLCV data + 150+ technical indicators
- **Pydantic v2 + pydantic-settings** — Config validation, hot-reloadable fields
- **aiosqlite** — SQLite persistence for budget tracking, trade history
- **structlog** — Structured JSON logging
- **aiolimiter** — Async rate limiting per OKX endpoint
- **python-dotenv** — Credentials from .env (never committed)
- **Docker** — Deployment target (Synology NAS / VPS)

## Architecture

```
Config (TOML + .env, Pydantic validated, hot-reload)
        |
OKX Exchange Client (python-okx wrapper, @retrier decorator, dry-run branch)
        |
Strategy Engine (IStrategy ABC, plugin loading, pandas-ta indicators)
        |  SignalResult
Risk Manager (BudgetManager + PositionSizer + ProtectionManager)
        |  TradeDecision (approved, size, SL, TP, time_limit)
Executor (place order + algo stop-loss on OKX, track lifecycle, report P&L)
```

### Key Design Principles (from competitive analysis)

These are drawn from analyzing Freqtrade (48.5k stars), Hummingbot (18k stars), and TradingAgents (49k stars):

1. **Triple Barrier on every position** — Stop-loss + take-profit + time-limit. Server-side stop-loss via OKX algo orders (executes even if bot crashes). Non-negotiable.
2. **Dry-run at the exchange layer** — `create_order()` branches to simulation or real execution. Uses real orderbook for slippage simulation. Separate SQLite DB for paper trades.
3. **Strategy-as-plugin** — Abstract `IStrategy` base class. Strategies loaded dynamically. Same code for backtest and live.
4. **Hard risk rules first, LLM second** — Budget limits, position sizing, drawdown thresholds are hard-coded rules. LLM interpretation is an optional enhancement layer, never the only safeguard.
5. **Retry with backoff on every exchange call** — `@retrier` decorator with exponential backoff (1, 4, 9, 16s). OKX has strict rate limits.
6. **Credential isolation** — Copy API keys to exchange config, scrub from main config dict to prevent accidental logging.

## Project Structure

```
neet-crypto-trader/
├── CLAUDE.md                       # This file
├── pyproject.toml                  # Dependencies, scripts, tool config
├── Dockerfile
├── docker-compose.yml
├── .env.example                    # Template: OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE
├── config/
│   └── default.toml                # Trading config (budget, pairs, strategy, risk)
├── src/
│   └── nct/                        # Main package
│       ├── __init__.py
│       ├── main.py                 # TradingAgent orchestrator, entry point
│       ├── config.py               # Pydantic settings (OKXCredentials, BudgetConfig, TradingConfig)
│       ├── exchange/
│       │   ├── __init__.py
│       │   ├── client.py           # OKXClient — wraps python-okx, @retrier, dry-run branch
│       │   ├── market_feed.py      # WebSocket real-time price stream (WsPublicAsync)
│       │   └── models.py           # Ticker, OrderRequest, OrderResponse, Position dataclasses
│       ├── strategy/
│       │   ├── __init__.py
│       │   ├── base.py             # IStrategy ABC: populate_indicators, populate_entry/exit_trend
│       │   ├── signals.py          # Signal enum (BUY/SELL/HOLD), SignalResult dataclass
│       │   └── momentum.py         # RSI + MACD momentum strategy
│       ├── risk/
│       │   ├── __init__.py
│       │   ├── budget_manager.py   # Weekly/monthly budget tracking, loss/gain limits (SQLite-backed)
│       │   ├── position_sizer.py   # Per-trade sizing: (balance * risk%) / stop_distance
│       │   ├── risk_manager.py     # Central gate — every trade passes through
│       │   └── protections.py      # StoplossGuard, MaxDrawdown, CooldownPeriod
│       ├── portfolio/
│       │   ├── __init__.py
│       │   └── tracker.py          # Open positions, P&L, sync with OKX
│       └── db.py                   # SQLite schema, migrations, persistence helpers
├── tests/
│   ├── conftest.py
│   ├── test_budget_manager.py
│   ├── test_position_sizer.py
│   ├── test_risk_manager.py
│   ├── test_strategy_momentum.py
│   └── test_exchange_client.py
└── scripts/
    └── backtest.py                 # Historical backtest runner
```

## Configuration

**Credentials** — `.env` file only, never in TOML or committed to git:
```
OKX_API_KEY=your-key
OKX_API_SECRET=your-secret
OKX_PASSPHRASE=your-passphrase
OKX_DEMO_MODE=true
```

**Trading config** — `config/default.toml`:
```toml
[trading]
pairs = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
strategy = "momentum"
timeframe = "15m"
max_open_positions = 3
paper_trading = true

[budget]
period = "weekly"
amount_usdt = 500
max_loss_pct = 5.0
max_gain_pct = 15.0
max_position_pct = 20.0
daily_loss_limit_usdt = 100

[risk]
stop_loss_pct = 3.0
take_profit_pct = 5.0
time_limit_seconds = 3600
trailing_stop = false
min_signal_confidence = 0.6
```

## Conventions

### Code Style
- Python 3.12+ features allowed (match statements, type unions with `|`)
- Use `Decimal` for all monetary values — never float
- Type hints on all public methods
- No docstrings on self-evident methods; add comments only where logic isn't obvious
- Ruff for linting + formatting (line-length = 100)

### Naming
- Package: `nct` (neet-crypto-trader)
- Config classes: `*Config` suffix (e.g., `BudgetConfig`, `TradingConfig`)
- Strategy classes: descriptive name + `Strategy` suffix (e.g., `MomentumStrategy`)
- Test files mirror source: `src/nct/risk/budget_manager.py` → `tests/test_budget_manager.py`

### Error Handling
- Exchange calls: `@retrier` decorator handles transient failures
- Strategy callbacks: wrap in `strategy_safe_wrapper` — log + continue, never crash the bot
- Budget/risk checks: return `(bool, reason_str)` tuples, never raise on denial

### Database
- SQLite via aiosqlite
- Tables: `trades`, `budget_periods`, `executor_log`
- Persist budget state so it survives restarts
- Separate DB files for live vs paper trading

### Testing
- pytest + pytest-asyncio
- Mock OKX API calls in exchange tests
- Test risk manager exhaustively (budget exhausted, daily limit, max positions, etc.)
- Use historical candle fixtures for strategy tests

## OKX-Specific Notes

- **Demo mode**: Set `flag="1"` in API calls for demo environment (fake funds)
- **Live mode**: Set `flag="0"` — requires explicit config change
- **Position mode**: Use "net mode" for spot, check long/short mode for futures
- **tdMode**: `"cash"` for spot, `"cross"`/`"isolated"` for futures
- **Rate limits**: ~20 req/s for market data, ~60 req/2s for order placement
- **Algo orders**: Use for server-side stop-loss/take-profit (survive bot crashes)
- **7-day order history limit**: Need archive endpoint for older closed orders
- **Broker ID**: Register for OKX broker program for higher rate limits (optional)

## Implementation Phases

1. **Foundation** — pyproject.toml, config system, OKX client, data models → can authenticate + fetch prices
2. **Risk Engine** — BudgetManager, PositionSizer, protections, SQLite persistence → budget enforcement
3. **Strategy Engine** — IStrategy ABC, momentum strategy, signal generation → buy/sell signals
4. **Execution** — Order placement, portfolio tracking, triple barrier → paper trading on OKX demo
5. **Main Loop** — TradingAgent orchestrator, scheduling, logging, graceful shutdown → autonomous operation
6. **Hardening** — Backtesting, Telegram notifications, Docker deployment, second strategy

## Deployment

- **Development**: Run locally (`python -m nct`)
- **Production**: Docker on Synology NAS or Hetzner VPS ($4/mo)
- Server-side stop-losses protect positions during bot downtime
- Telegram bot for monitoring + kill switch from phone
- `restart: always` in docker-compose for auto-recovery

## Safety Invariants (never violate)

1. Every position MUST have a server-side stop-loss (OKX algo order)
2. BudgetManager MUST be checked before every order — no bypass path
3. Paper trading (`demo_mode=true`) is the default — live requires explicit opt-in
4. API keys MUST never appear in logs, config files, or error messages
5. Withdraw permission MUST NOT be enabled on the OKX API key
6. The bot MUST gracefully close on SIGINT/SIGTERM (cancel open orders, log state)
