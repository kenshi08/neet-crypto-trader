"""Configuration loading and validation via Pydantic."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
def _find_config_path() -> Path:
    """Find the config file, checking common locations."""
    candidates = [
        Path('config/default.toml'),             # relative to cwd (Docker /app/)
        Path('/app/config/default.toml'),         # Docker absolute path
        Path(__file__).resolve().parent.parent.parent / 'config' / 'default.toml',  # dev
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]  # return first candidate even if not found


DEFAULT_CONFIG_PATH = _find_config_path()


# ---------------------------------------------------------------------------
# Credentials (from .env only — never in TOML)
# ---------------------------------------------------------------------------
class OKXCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='OKX_')

    api_key: str = ''
    api_secret: str = ''
    passphrase: str = ''
    demo_mode: bool = True
    base_url: str = 'https://www.okx.com'

    @property
    def flag(self) -> str:
        """OKX API flag: '1' for demo, '0' for live."""
        return '1' if self.demo_mode else '0'


# ---------------------------------------------------------------------------
# Bybit credentials (from .env only)
# ---------------------------------------------------------------------------
class BybitCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='BYBIT_')

    api_key: str = ''
    api_secret: str = ''
    demo_mode: bool = True  # uses testnet.bybit.com when True


# ---------------------------------------------------------------------------
# Coinbase credentials (from .env only)
# ---------------------------------------------------------------------------
class CoinbaseCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='COINBASE_')

    # CDP key name, e.g. "organizations/xxx/apiKeys/yyy"
    api_key: str = ''
    # PEM-formatted private key content (or path to key file)
    api_secret: str = ''
    # Coinbase Advanced Trade has no public sandbox. Demo mode = dry-run only.
    demo_mode: bool = True


# ---------------------------------------------------------------------------
# Trading configuration
# ---------------------------------------------------------------------------
class TradingConfig(BaseModel):
    pairs: list[str] = Field(default_factory=lambda: ['BTC-USDT', 'ETH-USDT', 'SOL-USDT'])
    strategy: str = 'momentum'
    timeframe: str = '15m'
    max_open_positions: int = 3
    poll_interval_seconds: int = 10

    @field_validator('pairs', mode='before')
    @classmethod
    def _validate_pairs(cls, v: list[str]) -> list[str]:
        for pair in v:
            if '-' not in pair:
                msg = f"Invalid pair format '{pair}' — expected 'BASE-QUOTE' (e.g. 'BTC-USDT')"
                raise ValueError(msg)
        return v

    @field_validator('timeframe')
    @classmethod
    def _validate_timeframe(cls, v: str) -> str:
        valid = {
            '1m', '3m', '5m', '15m', '30m',
            '1H', '2H', '4H', '6H', '12H',
            '1D', '1W', '1M',
        }
        if v not in valid:
            msg = f"Invalid timeframe '{v}' — must be one of {sorted(valid)}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# Budget configuration
# ---------------------------------------------------------------------------
class BudgetConfig(BaseModel):
    period: Literal['weekly', 'monthly'] = 'weekly'
    amount_usdt: Decimal = Decimal('500')
    max_loss_pct: Decimal = Decimal('5.0')
    max_gain_pct: Decimal = Decimal('15.0')
    max_position_pct: Decimal = Decimal('20.0')
    daily_loss_limit_usdt: Decimal = Decimal('100')

    @field_validator(
        'amount_usdt',
        'max_loss_pct',
        'max_gain_pct',
        'max_position_pct',
        'daily_loss_limit_usdt',
        mode='before',
    )
    @classmethod
    def _coerce_to_decimal(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ---------------------------------------------------------------------------
# Risk configuration
# ---------------------------------------------------------------------------
class RiskConfig(BaseModel):
    stop_loss_pct: Decimal = Decimal('3.0')
    take_profit_pct: Decimal = Decimal('5.0')
    time_limit_seconds: int = 3600
    trailing_stop: bool = False
    trailing_stop_activation_pct: Decimal = Decimal('2.0')
    trailing_stop_delta_pct: Decimal = Decimal('1.0')
    min_signal_confidence: float = 0.6

    @field_validator(
        'stop_loss_pct',
        'take_profit_pct',
        'trailing_stop_activation_pct',
        'trailing_stop_delta_pct',
        mode='before',
    )
    @classmethod
    def _coerce_to_decimal(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ---------------------------------------------------------------------------
# Telegram configuration (from .env only)
# ---------------------------------------------------------------------------
class TelegramConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='TELEGRAM_')

    # Credentials (from .env only)
    token: str = ''
    chat_id: str = ''

    # Alert filtering (from TOML)
    min_severity: str = 'low'

    # Quiet hours (from TOML) — suppress non-critical alerts during sleep
    quiet_hours_start: int = -1   # hour 0-23, -1 = disabled
    quiet_hours_end: int = -1
    quiet_hours_timezone: str = 'UTC'
    quiet_hours_min_severity: str = 'critical'

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)


# ---------------------------------------------------------------------------
# Top-level application config
# ---------------------------------------------------------------------------
class _ExchangeSelector(BaseSettings):
    """Reads the EXCHANGE env var to choose which exchange to use."""
    model_config = SettingsConfigDict(env_prefix='')

    exchange: Literal['okx', 'bybit', 'coinbase'] = 'okx'


class AppConfig(BaseModel):
    exchange: Literal['okx', 'bybit', 'coinbase'] = 'okx'
    okx: OKXCredentials = Field(default_factory=OKXCredentials)
    bybit: BybitCredentials = Field(default_factory=BybitCredentials)
    coinbase: CoinbaseCredentials = Field(default_factory=CoinbaseCredentials)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    strategy_params: dict = Field(default_factory=dict)


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from TOML file + environment variables.

    Credentials come from environment / .env file via OKXCredentials.
    Trading, budget, and risk settings come from the TOML config file.
    """
    path = config_path or DEFAULT_CONFIG_PATH

    toml_data: dict = {}
    if path.exists():
        import tomllib  # type: ignore[no-redef]

        with open(path, 'rb') as f:
            toml_data = tomllib.load(f)
        log.info('config_loaded', path=str(path))
    else:
        log.warning('config_file_not_found', path=str(path), using='defaults')

    okx = OKXCredentials()
    bybit = BybitCredentials()
    coinbase = CoinbaseCredentials()
    telegram_toml = toml_data.get('telegram', {})
    telegram = TelegramConfig(**telegram_toml)
    exchange_selector = _ExchangeSelector()

    trading = TradingConfig(**toml_data.get('trading', {}))
    budget = BudgetConfig(**toml_data.get('budget', {}))
    risk = RiskConfig(**toml_data.get('risk', {}))

    strategy_params = toml_data.get('strategy', {}).get(trading.strategy, {})

    config = AppConfig(
        exchange=exchange_selector.exchange,
        okx=okx, bybit=bybit, coinbase=coinbase,
        trading=trading, budget=budget, risk=risk,
        telegram=telegram, strategy_params=strategy_params,
    )

    creds_map = {'okx': config.okx, 'bybit': config.bybit, 'coinbase': config.coinbase}
    active_creds = creds_map[config.exchange]

    log.info(
        'config_summary',
        exchange=config.exchange,
        demo_mode=active_creds.demo_mode,
        pairs=config.trading.pairs,
        budget_period=config.budget.period,
        budget_amount=str(config.budget.amount_usdt),
        has_api_key=bool(active_creds.api_key),
    )

    return config
