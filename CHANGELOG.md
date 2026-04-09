# Changelog

All notable changes to neet-crypto-trader are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
