# neet-crypto-trader

Budget-controlled crypto trading agent supporting **Coinbase**, **OKX**, and **Bybit**. Automates short-term speculative trading with strict weekly/monthly budget limits on losses and gains.

## Features

- **Multi-exchange** — Switch between Coinbase, OKX, and Bybit via the `EXCHANGE` env var
- **Budget Controls** — Weekly/monthly capital limits, daily loss limits, daily notional cap
- **Risk Management** — Triple barrier on every position (SL + TP + time limit), atomic entry reversal, server-side stop-losses, correlation-aware position limits, volatility circuit breaker
- **Advanced Exits** — Trailing stop with activation threshold, breakeven stop move, partial profit taking at staged targets
- **Paper Trading** — Demo/testnet mode by default, with unambiguous mode announcement at startup
- **Realistic Backtests** — Fee-aware, slippage-modeled, with regime breakdown and per-pair contribution
- **Research Tooling** — Walk-forward validation, parameter stability grid search, lookahead bias checker
- **Pluggable Strategies** — Config-driven selection between RSI+MACD momentum and Bollinger Bands mean reversion
- **Market Selection** — Automatic pair filtering by volume, spread, and blacklist
- **Reconciliation** — Auto-fix stale trades, verify SL/TP barriers still exist, handle partial fills
- **Persistent State** — SQLite-backed budget tracking, trade analytics, command audit trail
- **Telegram Bot** — Inline keyboard UI, `/why` explainability, `/signal` indicators, `/stats` performance, `/close` per-position, alert severity with quiet hours

## Quick Start

### Prerequisites

- Python 3.11+
- One of:
  - **Coinbase** account — primary, accessible from Singapore ([create CDP API key](https://portal.cdp.coinbase.com/projects/api-keys))
  - **OKX** account ([create here](https://www.okx.com/account/my-api))
  - **Bybit** account ([live](https://www.bybit.com) or [testnet](https://testnet.bybit.com) — geo-blocked in some regions)
- API key with **Trade** permission only — **never** enable Withdraw

### Installation

```bash
git clone https://github.com/kenshi08/neet-crypto-trader.git
cd neet-crypto-trader
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

1. Copy the environment template and add credentials for your chosen exchange:
```bash
cp .env.example .env
# Edit .env — set EXCHANGE=coinbase|okx|bybit and fill in the matching keys
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

## Exchange Support

Behavior and guarantees differ by venue. Coinbase is the **primary** target
(what's tested end-to-end for Singapore); OKX and Bybit are alternative
backends with their own trade-offs.

| Capability              | Coinbase (primary)             | OKX                              | Bybit                          |
|-------------------------|--------------------------------|----------------------------------|--------------------------------|
| Singapore access        | Yes                            | Yes (with regional URL)          | Geo-blocked                    |
| Server-side stop-loss   | `stop_limit_order_gtc_*`       | Algo orders (`conditional`)      | V5 conditional (`triggerPrice`)|
| Server-side take-profit | `stop_limit_order_gtc_*`       | Algo orders (`conditional`)      | V5 conditional                 |
| Atomic SL/TP reversal   | Yes (#36)                      | Yes (#36)                        | Yes (#36)                      |
| Demo / sandbox          | Local dry-run (no public SB)   | OKX demo (`flag=1`)              | Testnet (`testnet.bybit.com`)  |
| Real balance in demo    | No (dry-run simulates)         | Yes (OKX demo is real balance)   | Yes (testnet coins)            |
| WebSocket market feed   | REST polling only              | Native WebSocket                 | REST polling only              |
| Typical taker fee       | ~0.4%                          | ~0.1%                            | ~0.1%                          |
| Quote currency          | USD / USDC                     | USDT                             | USDT                           |

**Notes:**

- **Triple barrier is enforced atomically on all three exchanges** — if
  stop-loss or take-profit placement fails after entry, `OrderExecutor`
  reverses the entry with a market order in the opposite direction (#36).
  This guarantee is exchange-agnostic.
- **Coinbase demo is local dry-run only**. Advanced Trade has no public
  sandbox, so orders are simulated in-process; public market data always
  uses the live endpoint so price action is real. Use `COINBASE_DEMO_MODE=true`.
- **OKX demo keys are separate from live keys** — they must be created while
  in "Demo Trading" mode on the OKX website, and only work against the demo
  environment (`flag=1`).
- **Market feed limitation**: only OKX has a WebSocket feed wired up;
  Coinbase and Bybit fall back to REST polling via `DataProvider`. This is
  fine for 15m timeframes but would matter at 1m or below.

## Project Structure

```
src/nct/
  config.py            # Pydantic config (TOML + .env)
  diagnostics.py       # DiagnosticEngine — trade explainability (/why, /signal)
  db.py                # SQLite: trades, budget, analytics, commands
  executor.py          # Order execution: triple barrier, trailing, breakeven, partial TP
  main.py              # Entry point / orchestrator
  notifier.py          # Telegram bot: inline UI, commands, alerts
  exchange/
    base.py            # IExchange ABC + get_algo_order_status
    coinbase_client.py # Coinbase Advanced Trade (primary)
    client.py          # OKXClient (python-okx)
    bybit_client.py    # BybitClient (pybit V5)
    models.py          # Typed data models (Ticker, Candle, Order, Position)
  strategy/
    base.py            # IStrategy ABC
    momentum.py        # RSI + MACD strategy
    mean_reversion.py  # Bollinger Bands + volume strategy
    data_provider.py   # OHLCV fetching, caching
    market_selector.py # Volume/spread/blacklist pair filter
  risk/
    budget_manager.py  # Budget tracking + daily notional cap
    risk_manager.py    # Central risk gate + correlation limits
    protections.py     # StoplossGuard, MaxDrawdown, CooldownPeriod, VolatilityCircuitBreaker
  portfolio/
    tracker.py         # Position tracking, P&L, reconciliation
  telegram/
    severity.py        # AlertSeverity enum, quiet hours
    keyboards.py       # Inline keyboard builders
scripts/
  backtest.py          # Backtest runner (fees, slippage, regime, per-pair)
  walk_forward.py      # Walk-forward validation
  param_stability.py   # Parameter grid search
  lookahead_check.py   # Lookahead bias detection
```

## Development

```bash
# Run tests (434 tests)
pytest tests/ -v

# Run linter
ruff check src/ tests/

# Run the bot (demo mode)
nct

# Backtesting
python scripts/backtest.py --pair BTC-USDT --timeframe 15m --limit 300
python scripts/backtest.py --pairs BTC-USDT ETH-USDT SOL-USDT --regime  # multi-pair + regime
python scripts/backtest.py --exchange okx                                 # 0.1% fees

# Research tools
python scripts/walk_forward.py --pair BTC-USDT --windows 5 --train-pct 70
python scripts/param_stability.py --param1 stop_loss_pct:1:5:0.5 --param2 take_profit_pct:3:8:1
python scripts/lookahead_check.py --strategy momentum

# Deploy with Docker
docker compose build --no-cache && docker compose up -d
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

All 12 phases are complete. See [GitHub Issues](https://github.com/kenshi08/neet-crypto-trader/issues) for details.

- **Phase 1-6** — Foundation, risk engine, strategies, execution, main loop, hardening
- **Phase 7** — Operator Intelligence: `/why`, `/signal`, `/close`, inline keyboard UI, alert severity (#44-#48, #68)
- **Phase 8** — Reconciliation & Resilience: auto-fix stale trades, verify barriers, partial fills, audit trail (#49-#52)
- **Phase 9** — Post-Trade Analytics: `trade_analytics` table, latency measurement, `/stats` command (#53-#55)
- **Phase 10** — Market Selection: volume/spread filters, correlation limits, notional cap, volatility breaker (#56-#59)
- **Phase 11** — Research Pipeline: walk-forward, parameter stability, regime breakdown, lookahead checker (#60-#64)
- **Phase 12** — Trailing & Staged Exits: trailing stop, breakeven move, partial profit taking (#65-#67)

## License

Private project.
