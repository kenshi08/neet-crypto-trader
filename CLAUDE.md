# CLAUDE.md — neet-crypto-trader

## Project Overview

A crypto trading agent for OKX that automates short-term speculative trading with strict budget controls (weekly/monthly limits on losses and gains). Designed for a single user managing a personal trading budget.

## Tech Stack

- **Python 3.11+**
- **python-okx** — Official OKX SDK (REST + WebSocket)
- **pybit** — Official Bybit SDK (V5 unified trading API)
- **pandas + ta** — OHLCV data + technical indicators (ta library; pandas-ta requires 3.12+)
- **Pydantic v2 + pydantic-settings** — Config validation, hot-reloadable fields
- **aiosqlite** — SQLite persistence for budget tracking, trade history
- **structlog** — Structured JSON logging
- **aiolimiter** — Async rate limiting per exchange endpoint
- **python-dotenv** — Credentials from .env (never committed)
- **Docker** — Deployment target (Synology NAS / VPS)

## Architecture

```
Config (TOML + .env, EXCHANGE selector, Pydantic validated)
        |
Exchange Factory → IExchange (OKXClient | BybitClient)
        |  (@retrier decorator, dry-run branch, server-side SL/TP)
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

## Git Practices

### Branch Strategy

This project uses **GitLab Flow** — a simplified branching model suited for a single-developer project with CI/CD.

- **`main`** — Production-ready code. Always deployable. Protected branch.
- **`develop`** — Integration branch. Features merge here first, then promote to `main` via merge request.
- **Feature branches** — `feature/<short-description>` (e.g., `feature/budget-manager`, `feature/momentum-strategy`)
- **Fix branches** — `fix/<short-description>` (e.g., `fix/okx-rate-limit-retry`)
- **Chore branches** — `chore/<short-description>` (e.g., `chore/update-dependencies`)

### Branch Naming

```
feature/add-trailing-stop      # New functionality
fix/budget-reset-timezone       # Bug fix
chore/upgrade-pandas-ta         # Maintenance, deps, config
refactor/extract-order-executor # Code restructure, no behavior change
docs/deployment-guide           # Documentation only
test/risk-manager-edge-cases    # Test additions only
```

- Use lowercase, kebab-case
- Keep names short but descriptive (3-5 words max)
- Never commit directly to `main` or `develop` — always use a branch + merge request

### Commit Messages

Follow **Conventional Commits** format:

```
<type>(<scope>): <short summary>

<optional body — explain WHY, not WHAT>

<optional footer — breaking changes, issue refs>
```

**Types:**
- `feat` — New feature (e.g., `feat(strategy): add RSI divergence detection`)
- `fix` — Bug fix (e.g., `fix(budget): reset daily counter at UTC midnight`)
- `refactor` — Code change that neither fixes a bug nor adds a feature
- `test` — Adding or updating tests
- `chore` — Build, deps, CI, tooling (e.g., `chore(deps): upgrade python-okx to 5.4`)
- `docs` — Documentation only
- `perf` — Performance improvement
- `ci` — CI/CD pipeline changes

**Scopes** (match project modules):
- `exchange`, `strategy`, `risk`, `portfolio`, `config`, `db`, `main`, `deps`, `docker`

**Rules:**
- Imperative mood in summary: "add feature" not "added feature" or "adds feature"
- Summary under 72 characters
- Body wraps at 100 characters
- Reference GitLab issues when applicable: `Closes #12` or `Relates to #34`
- One logical change per commit — don't mix refactoring with features

**Examples:**
```
feat(risk): add weekly budget reset with timezone support

Budget periods now reset based on configurable timezone (default UTC).
The reset timestamp is persisted to SQLite so it survives restarts.

Closes #15
```

```
fix(exchange): handle OKX rate limit 429 with exponential backoff

OKX returns HTTP 429 when rate limits are exceeded. The @retrier
decorator now catches this specifically and uses quadratic backoff
(1, 4, 9, 16 seconds) before retry.
```

```
test(budget): add edge cases for mid-period restart recovery
```

### Commit Hygiene

- **Atomic commits** — Each commit should compile, pass tests, and represent one logical change
- **No WIP commits on shared branches** — Squash or reword before merging to `develop`
- **Never commit secrets** — `.env`, API keys, private keys. If accidentally committed, rotate the key immediately (git history is permanent)
- **Never commit generated files** — `__pycache__/`, `.pyc`, `.egg-info/`, `*.sqlite` (except schema migrations)
- **Sign commits** if GPG is configured (optional but recommended)

### Merge Requests (GitLab)

- Title follows commit convention: `feat(strategy): add Bollinger Band mean reversion`
- Description includes: what changed, why, how to test
- Self-review the diff before marking ready
- Squash merge to `develop` to keep history clean
- Delete source branch after merge

### Tags & Releases

- Use semantic versioning: `v0.1.0`, `v0.2.0`, `v1.0.0`
- Tag on `main` after promoting from `develop`
- `v0.x.x` — Pre-production (paper trading validation)
- `v1.0.0` — First live trading release

### .gitignore Essentials

The `.gitignore` must include:
```
# Secrets
.env
*.pem
*.key

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/
venv/

# Data (generated at runtime)
*.sqlite
*.sqlite-journal
data/
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
docker-compose.override.yml
```

## Issue Tracking — Bug & Problem Management

### When to Create a Bug Issue

During development, debugging, and troubleshooting, create a GitHub issue when:

- **A bug required real investigation** — not a typo, but a problem that took debugging to understand
- **OKX API behavior differs from documentation** — undocumented quirks, unexpected responses, rate limit nuances
- **Dependency incompatibility or breaking change** — version conflicts, missing features, platform-specific failures
- **Edge case discovered that needs architectural change** — not just a test fix, but a design gap
- **Data integrity risk** — anything that could cause incorrect P&L calculation, budget tracking errors, or missed stop-losses
- **Production-impacting issue** — even if quickly fixed, document it for the post-mortem record

### When NOT to Create an Issue (just fix inline)

- Typos, lint errors, formatting fixes
- Obvious mock/test setup mistakes
- Config value adjustments
- Import ordering, unused variable cleanup

### Bug Issue Workflow

```
1. Hit a problem during development
2. Assess: trivial fix or real investigation?
   ├── Trivial → fix inline, no issue needed
   └── Significant → continue to step 3
3. Create issue BEFORE fixing (captures the raw problem state):
   - Title: `bug(<scope>): <concise description>`
   - Label: `bug` + relevant phase label
   - Body: symptoms, root cause (if known), environment details
4. Fix the bug
5. Commit referencing the issue: `fix(exchange): handle empty ticker response — Closes #25`
6. Close the issue (automatically via commit, or manually with resolution notes)
```

### Bug Issue Template

```markdown
## Bug: <title>

**Discovered during:** Phase X — <what you were building/testing>
**Severity:** critical | major | minor
**Component:** exchange | risk | strategy | config | db

### Symptoms
What went wrong. Error messages, unexpected behavior, test failures.

### Root Cause
Why it happened. The actual underlying problem.

### Environment
- Python version, OS, dependency versions if relevant
- OKX demo vs live mode
- Config values that trigger the issue

### Fix
What was changed to resolve it. Reference the commit.

### Prevention
How to prevent similar issues. New test added? Validation rule? Documentation update?
```

### Severity Guide

| Severity | Definition | Example |
|---|---|---|
| **critical** | Could cause financial loss, data corruption, or safety invariant violation | Stop-loss not placed, budget check bypassed, wrong order size |
| **major** | Breaks core functionality, requires investigation, blocks progress | OKX API auth failing silently, candle data gaps, retry not triggering |
| **minor** | Inconvenience, workaround exists, non-blocking | Log formatting issue, slow test, config error message unclear |

### Labels for Bug Issues

- `bug` — always present
- `phase:N-*` — which phase it was discovered in
- `critical` — for severity:critical bugs (financial risk)
- `okx-quirk` — for OKX API documentation gaps or undocumented behavior

### Post-Resolution

After closing a bug issue, consider:
1. **Add a test** that would have caught this bug
2. **Update CLAUDE.md** if it reveals a new convention or OKX-specific note
3. **Update config validation** if bad config caused the bug
4. **Add to OKX-Specific Notes** section if it's an exchange quirk

This builds a knowledge base — when a similar problem appears later, the closed issue provides context and the fix approach.

## Coding Conventions

### Python Style

- **Python 3.11+** features allowed: `match` statements, `X | Y` unions, `StrEnum`
- **Ruff** for linting + formatting — single tool, no black/isort/flake8 needed
  - Line length: 100
  - Target: `py311`
  - Select: `["E", "F", "W", "I", "N", "UP", "B", "A", "SIM", "RUF"]`
- **No trailing whitespace**, UTF-8 encoding, LF line endings
- **Imports**: stdlib → third-party → local, separated by blank lines (Ruff `I` handles this)
- Single quotes for strings unless the string contains a single quote

### Type Safety

- Type hints on **all** public methods and module-level functions
- Private helpers: type hints recommended but not required
- Use `Decimal` for all monetary values — **never float** for prices, balances, or P&L
- Use `datetime` (timezone-aware, UTC) for all timestamps — never naive datetimes
- Prefer `dataclass` or Pydantic `BaseModel` over raw dicts for structured data
- Use `Literal` types for fixed string options: `Literal["buy", "sell"]` not `str`
- Use `TypeAlias` for complex types: `PairWithTimeframe: TypeAlias = tuple[str, str]`

### Function & Method Design

- Functions should do one thing
- Max function length: ~40 lines (if longer, extract a helper)
- Prefer returning values over mutating arguments
- Use keyword-only arguments for functions with 3+ parameters: `def place_order(*, pair, side, size)`
- Default to immutable: `tuple` over `list` for return types when mutation isn't needed
- Async functions: use `async def` only when actually awaiting I/O — don't make things async needlessly

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Package | lowercase, short | `nct` |
| Module | snake_case | `budget_manager.py` |
| Class | PascalCase | `BudgetManager` |
| Function/Method | snake_case | `calculate_position_size()` |
| Constant | UPPER_SNAKE | `MAX_RETRY_COUNT = 4` |
| Private | leading underscore | `_validate_budget()` |
| Config class | `*Config` suffix | `TradingConfig` |
| Strategy class | `*Strategy` suffix | `MomentumStrategy` |
| ABC/Interface | `I*` or `*Base` prefix/suffix | `IStrategy`, `ExecutorBase` |
| Enum | PascalCase class, UPPER members | `Signal.BUY` |
| Type alias | PascalCase | `TradeDecision` |
| Test file | `test_` prefix, mirrors source | `test_budget_manager.py` |
| Test function | `test_` prefix, descriptive | `test_budget_rejects_when_weekly_limit_exceeded()` |

### Error Handling

- **Exchange calls**: `@retrier` decorator handles transient failures (network, rate limits)
- **Strategy callbacks**: wrap in `strategy_safe_wrapper` — log + continue, never crash the bot
- **Budget/risk checks**: return `tuple[bool, str]` — `(approved, reason)`, never raise on denial
- **Expected errors**: handle explicitly with specific exception types
- **Unexpected errors**: let them propagate to the main loop's catch-all, log with full traceback
- **Never silence exceptions** with bare `except:` or `except Exception: pass`
- **Custom exceptions** inherit from a project base: `class NCTError(Exception)` → `class ExchangeError(NCTError)` → `class RateLimitError(ExchangeError)`

### Exception Hierarchy

```python
NCTError                        # Base for all project exceptions
├── ConfigError                 # Invalid configuration
├── ExchangeError               # OKX API errors
│   ├── RateLimitError          # 429 / rate limit exceeded
│   ├── AuthenticationError     # Invalid API key / signature
│   └── OrderError              # Order rejected / insufficient funds
├── StrategyError               # Strategy computation failure
├── BudgetExhaustedError        # Budget limit reached (informational, not crash-worthy)
└── DataError                   # Missing/corrupt market data
```

### Logging

- Use `structlog` for structured JSON logging
- Log levels:
  - `DEBUG` — Indicator values, raw API responses (development only)
  - `INFO` — Trade opened/closed, budget status, strategy signals
  - `WARNING` — Rate limit hit (retrying), position near stop-loss, budget approaching limit
  - `ERROR` — Order failed, API error after retries exhausted, data feed disconnected
  - `CRITICAL` — Kill switch triggered, budget limit breached, unrecoverable state
- Always include context: `log.info("order_placed", pair=pair, side=side, size=size, price=price)`
- **Never log**: API keys, secrets, passphrases, full account balances in production
- Separate log files: `trades.log` (trade events only), `agent.log` (general), `errors.log` (WARNING+)

### Documentation

- **No docstrings** on self-evident methods (`get_balance()`, `cancel_order()`)
- **Do add docstrings** on non-obvious logic, complex algorithms, or public API boundaries
- Use Google-style docstrings when needed:
  ```python
  def calculate_position_size(
      self, balance: Decimal, price: Decimal, stop_distance: Decimal
  ) -> Decimal:
      """Size position so max loss equals risk_per_trade % of balance.

      Args:
          balance: Available trading capital in USDT.
          price: Current asset price.
          stop_distance: Distance to stop-loss as a decimal (e.g., 0.03 for 3%).

      Returns:
          Position size in base currency units.
      """
  ```
- Inline comments: explain **why**, not **what** — the code shows what
- TODO format: `# TODO(username): description — #issue-number`

### README.md

The `README.md` is the public face of the project. Keep it updated when:
- A new major feature is added (new phase completed)
- Setup instructions change (new dependency, config change)
- Project structure changes (new module or directory)
- Deployment instructions change

Sections to maintain: Quick Start, Configuration Reference, Project Structure, Development, Safety, Roadmap.

### CHANGELOG.md

Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. **Update on every commit to `develop` or `main`.**

**When to update:**
- Every phase completion — summarize what was added
- Every bug fix — note what was fixed and the root cause
- Every dependency change — note what changed and why
- Every breaking config change — note the migration path

**Categories:**
- `Added` — new features, new modules, new config options
- `Changed` — changes to existing functionality
- `Fixed` — bug fixes
- `Removed` — removed features or deprecated code
- `Security` — vulnerability fixes

**Workflow:**
1. During development, add entries under `## [Unreleased]`
2. When tagging a release, move unreleased entries to a versioned section `## [0.2.0] - YYYY-MM-DD`
3. Every entry should be a single line, human-readable, linking to the issue when relevant

**Example:**
```markdown
## [Unreleased]

### Added
- BudgetManager with weekly/monthly budget tracking and SQLite persistence (#4)
- PositionSizer with risk-based per-trade sizing (#5)

### Fixed
- OKX rate limit handling now retries on HTTP 429 with quadratic backoff (#23)
```

### Testing

- **Framework**: pytest + pytest-asyncio
- **Structure**: test files mirror source — `src/nct/risk/budget_manager.py` → `tests/test_budget_manager.py`
- **Naming**: `test_<method>_<scenario>_<expected>()` — e.g., `test_can_open_trade_returns_false_when_weekly_limit_exceeded()`
- **Mock external I/O**: Always mock OKX API calls, never hit real endpoints in tests
- **Fixtures**: Use `conftest.py` for shared fixtures (sample candles, mock exchange responses)
- **Coverage targets**: Risk module — 95%+, Exchange module — 80%+, Strategy — 80%+
- **Test categories**:
  - Unit tests: fast, isolated, no I/O
  - Integration tests: marked with `@pytest.mark.integration`, may use SQLite
  - Slow tests: marked with `@pytest.mark.slow`, for backtesting validation
- **Run before commit**: `pytest tests/ -x -q` (fail fast, quiet output)
- **Edge cases to always test in risk module**:
  - Budget exactly at limit
  - Budget exceeded by 1 unit
  - Mid-period restart recovery
  - Concurrent position limit reached
  - Zero balance
  - Negative P&L exceeding daily loss limit

### Dependencies

- Pin major+minor versions in `pyproject.toml`: `python-okx>=5.3,<6`
- Use `uv` or `pip-compile` for reproducible lock files
- Minimize dependencies — every dependency is a liability for a financial system
- Audit new dependencies before adding: check maintenance status, security history, license

## Exchange-Specific Notes

### Bybit (recommended)

- **Testnet**: `testnet.bybit.com` — completely separate environment from production. Set `BYBIT_DEMO_MODE=true` to use it. The `pybit` SDK switches automatically via the `testnet=True` constructor flag.
- **Testnet funds**: Request via testnet UI — instantly credited (no maintenance windows).
- **Symbols**: Bybit uses `BTCUSDT` (no dash). Our `_to_bybit_symbol()` converts from `BTC-USDT`.
- **Timeframes**: Bybit uses `1`, `15`, `60`, `240`, `D` instead of `1m`, `15m`, `1H`, `4H`, `1D`. Our `_to_bybit_timeframe()` handles this.
- **Categories**: `spot`, `linear` (USDT perpetual), `inverse` (coin-margined), `option`. We default to `spot`.
- **Account type**: `UNIFIED` for V5 API (default in our BybitClient).
- **Conditional orders**: Stop-loss/take-profit via `place_order` with `triggerPrice` + `triggerDirection` (1=rises above, 2=falls below).
- **Rate limits**: ~20 req/s for most endpoints (more generous than OKX).
- **Error codes**: `10003` invalid key, `10018` rate limit, `110001` order failed.

### OKX

- **Demo mode**: Set `flag="1"` in API calls for demo environment (fake funds)
- **Live mode**: Set `flag="0"` — requires explicit config change
- **Position mode**: Use "net mode" for spot, check long/short mode for futures
- **tdMode**: `"cash"` for spot, `"cross"`/`"isolated"` for futures
- **Rate limits**: ~20 req/s for market data, ~60 req/2s for order placement
- **Algo orders**: Use for server-side stop-loss/take-profit (survive bot crashes)
- **7-day order history limit**: Need archive endpoint for older closed orders
- **Broker ID**: Register for OKX broker program for higher rate limits (optional)
- **Regional endpoints** (#26): OKX uses different base URLs by region. API keys from one domain don't work on another:
  - Global: `https://www.okx.com`
  - Europe (EEA): `https://my.okx.com`
  - US: `https://app.okx.com`
  - Set via `OKX_BASE_URL` env var. The python-okx SDK `domain` parameter passes this to all API calls.
- **Demo API keys are separate from live keys**: Must be created while in "Demo Trading" mode on OKX website. Error `50119` usually means wrong domain or demo/live key mismatch.
- **Demo account service unreliability** (#26, #28): OKX EEA demo (`my.okx.com`) frequently returns 503/50001 on authenticated endpoints. The bot is resilient but Bybit testnet is more reliable for development.

### Adding a new exchange

1. Create `src/nct/exchange/<name>_client.py` implementing `IExchange`
2. Add `<Name>Credentials(BaseSettings)` in `src/nct/config.py`
3. Add the new exchange to the `Literal` type and the `_ExchangeSelector`
4. Update `create_exchange_client()` in `src/nct/exchange/factory.py`
5. Write tests for the new client (mirror `test_bybit_client.py`)
6. Update `.env.example` and CLAUDE.md

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
