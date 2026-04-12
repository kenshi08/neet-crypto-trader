# Changelog

All notable changes to neet-crypto-trader are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 12: Trailing & Staged Exits (#65-#67)
- Trailing stop activation after minimum gain — ratchets SL behind high-water mark
- Breakeven stop move — moves SL to entry + fees once profit exceeds trigger
- Partial profit taking — closes configurable fractions at staged profit targets
- Unified `check_exit_management()` called every iteration for open positions

### Added — Phase 11: Research Pipeline (#60-#64)
- Walk-forward validation script (`scripts/walk_forward.py`) — N-window train/test splits with overfitting verdict
- Parameter stability grid search (`scripts/param_stability.py`) — 2D sweep outputting CSV
- Regime-tagged performance breakdown in backtest report — trending/ranging/down metrics
- Lookahead bias checker (`scripts/lookahead_check.py`) — detects forward-looking mistakes
- Per-pair contribution report — multi-pair backtest with `--pairs` flag

### Added — Phase 10: Market Selection & Portfolio Intelligence (#56-#59)
- Market selector module — filters pairs by 24h volume, bid-ask spread, and blacklist
- Correlation-aware position limits — configurable groups prevent concentrated exposure
- Daily notional cap in BudgetManager — hard limit on total entry notional per day
- Volatility circuit breaker — halts entries when ATR exceeds N x median ATR

### Added — Phase 9: Post-Trade Analytics (#53-#55)
- `trade_analytics` DB table — per-trade execution quality (slippage, fill rate, latency, fees)
- Execution latency measurement — `time.monotonic()` instrumentation on all exchange calls
- `/stats` Telegram command — aggregate performance with period filters and per-pair breakdown

### Added — Phase 8: Reconciliation & Resilience (#49-#52)
- Reconciliation auto-fix — stale trades auto-closed, untracked positions detected with CRITICAL alert
- SL/TP barrier verification — periodic check that algo orders still exist, auto-repair or close
- Partial fill handling — uses actual `filled_size` for SL/TP sizing, handles zero fills
- Telegram command audit trail — all commands logged to `telegram_commands` DB table
- `get_algo_order_status()` added to IExchange and all 3 exchange clients

### Added — Phase 7: Operator Intelligence (#44-#48, #68)
- `/why <pair>` command — runs signal + risk pipeline, shows per-gate pass/fail
- `/signal <pair>` command — shows current indicator values (RSI, MACD, ATR, EMA)
- `/close <pair>` command — closes a single position with P&L reporting
- `/start` inline keyboard UI — button-based navigation, position cards, confirmations
- Alert severity levels (low/medium/high/critical) with quiet hours filtering
- `DiagnosticEngine` — reusable explainability engine for /why and /signal
- `telegram/severity.py` and `telegram/keyboards.py` subpackage

### Fixed
- **CRITICAL: Stop-loss atomicity** — `OrderExecutor` now reverses the entry
  if stop-loss or take-profit placement fails, eliminating the possibility
  of an unprotected open position. Previously the code logged a CRITICAL
  warning and kept the position open, violating the safety invariant. (#36)
- **Strategy selection is now config-driven** — Previously `main.py` hardcoded
  `MomentumStrategy` even though `MeanReversionStrategy` was available. New
  strategy factory (`src/nct/strategy/factory.py`) reads `config.trading.strategy`
  and instantiates the correct class. Setting `strategy = "mean_reversion"` in
  TOML now actually uses that strategy. (#37)
- **Backtest now models exchange fees** — `run_backtest()` previously reported
  gross P&L, making Coinbase strategies look ~0.8% per round-trip more
  profitable than reality. Each trade now tracks `entry_fee`, `exit_fee`,
  `gross_pnl`, and net `pnl`. The report shows gross/fees/net breakdown, and
  the CLI accepts `--fee-pct` or `--exchange {coinbase,okx,bybit,binance}`
  presets. Default is 0.4% per side (Coinbase Advanced Trade taker). (#38)
- **Running mode is now unambiguous** — Logs and Telegram previously mixed
  paper, demo, and live indicators together, making it easy to forget which
  mode was active. The bot now emits a single `running_mode` log line at
  startup with one of `PAPER_DRY_RUN`, `DEMO_REAL_BALANCE`, or
  `LIVE_REAL_MONEY`, and sends a matching Telegram message. Live mode is
  visually distinct (bold, warning text). (#39)
- **Backtest now models execution realism** — Previously entries filled at
  the exact candle close, stop-losses at the exact trigger, and take-profits
  at the exact level — none of which reflects real market-order execution.
  `run_backtest()` now applies adverse slippage (default 0.05% per side) to
  entries, signal exits, and stop-losses past the trigger. Stop-losses model
  gap-through: if a candle opens below the stop, the fill uses the open
  price (not the trigger), realistically capturing overnight gaps and flash
  crashes. Take-profits still fill at the exact level (limit-order semantics).
  CLI accepts `--slippage-pct` for explicit control.

### Added
- **Backtest realism filters** — next layer of execution realism: min-notional
  rejection, synthetic bid-ask spread with half-spread cost on market
  entries/exits, `max_spread_pct` filter, seeded random order rejection
  (stress-tests the retry path), and partial-fill approximation scaled by
  size/candle-volume ratio. All filters default to a strict no-op. CLI flags:
  `--min-notional`, `--spread-pct`, `--max-spread-pct`, `--rejection-rate`,
  `--rng-seed`, `--partial-fill-impact`. Rejection counters appear in the
  backtest report. Exchange presets unified into `_EXCHANGE_PRESETS` carrying
  both fee and min-notional values. 12 new tests (314 total). (#42)
- Strategy factory pattern with `create_strategy()` and `list_strategies()` (#37)
- Backtest fee model with per-side fee configuration and exchange presets (#38)
- Backtest slippage model with gap-through stop-loss semantics and
  `--slippage-pct` CLI flag (default 0.05% for liquid pairs)
- `src/nct/runtime_mode.py` — `RunningMode` enum, `detect_mode()`, and
  `describe()` with mode-specific log warnings and Telegram messages (#39)
- 22 new tests covering fee model, runtime mode detection, and slippage
  model including gap-through validation
- `config/default.toml` now includes both `[strategy.momentum]` and
  `[strategy.mean_reversion]` sections with documented parameters
- README documents how to switch strategies via config
- **Multi-exchange support: Coinbase** — third exchange option, accessible from Singapore (#33, #34, #35)
  - `CoinbaseClient` implementing `IExchange` via `coinbase-advanced-py` SDK
  - `CoinbaseCredentials` config with `COINBASE_` env prefix (CDP API keys)
  - Dry-run simulation for demo mode (Advanced Trade has no public sandbox)
  - Stop-loss/take-profit via `stop_limit_order_gtc_*` with `stop_direction`
  - Granularity mapping: `1m`/`15m`/`1H`/`1D` → `ONE_MINUTE`/`FIFTEEN_MINUTE`/`ONE_HOUR`/`ONE_DAY`
  - 23 new tests (269 total) for CoinbaseClient and factory
- **Multi-exchange support: Bybit** — Switch between OKX and Bybit via `EXCHANGE` env var
  - `IExchange` abstract interface in `src/nct/exchange/base.py` (#29)
  - `BybitClient` implementation using `pybit` SDK with V5 unified trading API (#30)
  - `BybitCredentials` config with `BYBIT_` env prefix
  - `create_exchange_client()` factory selecting between OKX and Bybit (#31)
  - Symbol/timeframe conversion (`BTC-USDT` ↔ `BTCUSDT`, `15m` ↔ `15`)
  - Bybit conditional orders for server-side stop-loss/take-profit
  - Bybit testnet support via `BYBIT_DEMO_MODE=true`
  - 24 new tests for Bybit client and exchange factory (246 total)
  - `.env.example` updated with comprehensive multi-exchange template
  - `CLAUDE.md` updated with Bybit notes and "Adding a new exchange" guide

### Added
- **Phase 6: Hardening** — Backtesting, Telegram bot, Docker deployment, second strategy
  - Backtesting engine with SL/TP simulation, Sharpe ratio, max drawdown, win rate metrics (#18)
  - Telegram bot — trade notifications, `/status`, `/stop` kill switch, `/balance` commands (#19)
  - Docker deployment — multi-stage Dockerfile, docker-compose with Synology NAS support (#20)
  - `MeanReversionStrategy` — Bollinger Bands + volume confirmation for ranging markets (#21)
  - OKX regional base URL support (`OKX_BASE_URL`) for EEA (`my.okx.com`) and US (`app.okx.com`)
  - TOML-configurable strategy parameters via `[strategy.momentum]` section (#27)
  - `NCT_CONFIG` env var to select alternate config files without rebuilding
  - `config/aggressive_test.toml` for demo account testing

### Fixed
- Docker: permission error on logs/data directories when running as non-root user (#24)
- Logging: graceful fallback to console-only when log directory isn't writable (#24)
- Config: find `default.toml` correctly across Docker and dev environments (#25)
- Exchange: OKX regional endpoint mismatch causing error 50119 — API key doesn't exist (#26)
- Exchange: bot crashes on OKX 503 and misclassifies error 50001 as auth failure (#28)
- Exchange: bot now starts even if OKX is temporarily unreachable — retries in trading loop (#28)
- **Phase 5: Main Loop** — TradingAgent orchestrator, graceful shutdown, structured logging
  - `TradingAgent` orchestrator wiring all components into autonomous trading loop (#15)
  - Graceful shutdown on SIGINT/SIGTERM — cancels orders, persists state, closes DB (#16)
  - Kill switch — emergency close all positions at market (#16)
  - `AgentState` state machine: STARTING → RUNNING → STOPPING → STOPPED
  - Structured logging with `agent.log`, `trades.log`, `errors.log` + rotation (#17)
  - Trading iteration: budget check → time-limit expiry → fetch candles → strategy → risk → execute
- **Phase 4: Execution** — Order executor, portfolio tracker, and WebSocket market feed
  - `OrderExecutor` — places orders with triple barrier (SL + TP + time limit) (#12)
  - `PortfolioTracker` — tracks open trades, P&L, persists to SQLite, detects orphans on startup (#13)
  - `MarketFeed` — WebSocket real-time ticker streaming with auto-reconnect (#14)
  - `TrackedTrade` dataclass with unrealized P&L calculation
  - Time-limit expiry checking and automatic position closure
- **Phase 3: Strategy Engine** — Strategy interface, momentum strategy, and data pipeline
  - `IStrategy` ABC with `populate_indicators`, `populate_entry_trend`, `populate_exit_trend` (#9)
  - `Signal` enum (BUY/SELL/HOLD) and `SignalResult` dataclass with confidence and SL/TP (#9)
  - `strategy_safe_wrapper` decorator — catches strategy exceptions, logs, continues (#9)
  - `MomentumStrategy` — RSI + MACD crossover with ATR-based dynamic SL/TP (#10)
  - `DataProvider` — OHLCV fetching with TTL cache, candle-to-DataFrame conversion (#11)
  - `candles_to_dataframe()` — converts OKX `Candle` objects to pandas DataFrames (#11)
- **Phase 2: Risk Engine** — Budget management, position sizing, and protection plugins
  - `BudgetManager` with weekly/monthly budget tracking, loss/gain limits, SQLite persistence (#4)
  - `PositionSizer` with risk-based per-trade sizing, confidence scaling, budget caps (#5)
  - `RiskManager` central gate orchestrating all risk checks before every trade (#6)
  - `StoplossGuard` — locks trading after N consecutive stop-losses within lookback period (#7)
  - `MaxDrawdown` — stops trading when drawdown from high-water mark exceeds threshold (#7)
  - `CooldownPeriod` — enforces wait time between trades on same pair (#7)
  - `ProtectionManager` chaining all protection plugins (#7)
  - `Database` async SQLite layer with trades, budget_periods, daily_stats tables (#8)
  - Extracted exception hierarchy to `src/nct/exceptions.py` for cross-module use
- `README.md` with quick start, configuration reference, project structure, safety notes
- `CHANGELOG.md` following Keep a Changelog format

## [0.1.0] - 2026-04-09

### Added
- **Phase 1: Foundation** — Project scaffold, configuration system, and OKX exchange client
- `pyproject.toml` with all dependencies (`python-okx`, `pandas`, `ta`, `pydantic`, `structlog`)
- Pydantic configuration system loading from TOML + `.env` with full validation
- `OKXClient` — OKX REST API wrapper with:
  - `@retrier` decorator (exponential backoff: 1, 4, 9, 16s)
  - Per-endpoint async rate limiters
  - Dry-run order simulation for paper trading
  - Server-side stop-loss/take-profit via OKX algo orders
- Typed data models: `Ticker`, `Candle`, `OrderRequest`, `OrderResponse`, `Position`, `AccountBalance`
- Factory parsers for raw OKX API responses
- `config/default.toml` with trading, budget, and risk configuration
- `.env.example` with OKX credential template
- `.gitignore` covering secrets, Python artifacts, runtime data
- `CLAUDE.md` with architecture, Git practices, coding conventions, bug tracking workflow
- 59 tests covering config validation, retry logic, response parsing, dry-run simulation

### Fixed
- `python-okx` version pinned to `>=0.3,<1` (PyPI uses `0.x.x` scheme, not `5.x`)
- Replaced `pandas-ta` with `ta` library (pandas-ta requires Python 3.12+)
- Python requirement lowered from 3.12 to 3.11 to match runtime environment

[Unreleased]: https://github.com/kenshi08/neet-crypto-trader/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kenshi08/neet-crypto-trader/releases/tag/v0.1.0
