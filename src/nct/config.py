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
# Hyperliquid credentials (from .env only)
# ---------------------------------------------------------------------------
class HyperliquidCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='HL_')

    # Wallet private key (hex, with or without 0x prefix)
    private_key: str = ''
    # Testnet mode (uses api.hyperliquid-testnet.xyz)
    demo_mode: bool = True
    # Optional vault address for vault-based trading
    vault_address: str = ''

    @property
    def api_key(self) -> str:
        """Compatibility: treat private_key as api_key for mode detection."""
        return self.private_key


# ---------------------------------------------------------------------------
# Trading configuration
# ---------------------------------------------------------------------------
class TradingConfig(BaseModel):
    pairs: list[str] = Field(default_factory=lambda: ['BTC-USDT', 'ETH-USDT', 'SOL-USDT'])
    strategy: str = 'momentum'
    timeframe: str = '15m'
    max_open_positions: int = 3
    poll_interval_seconds: int = 10
    reconciliation_interval: int = 10  # run reconciliation every N iterations
    confirmation_timeframes: list[str] = Field(default_factory=list)  # e.g. ["1H", "4H"]

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
    daily_notional_cap_usdt: Decimal = Decimal('0')  # 0 = disabled

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
class MarketSelectionConfig(BaseModel):
    min_volume_usdt: Decimal = Decimal('0')
    max_spread_pct: Decimal = Decimal('0')
    blacklist: list[str] = Field(default_factory=list)

    @field_validator('min_volume_usdt', 'max_spread_pct', mode='before')
    @classmethod
    def _coerce_to_decimal(cls, v: object) -> Decimal:
        return Decimal(str(v))


class RiskConfig(BaseModel):
    stop_loss_pct: Decimal = Decimal('3.0')
    take_profit_pct: Decimal = Decimal('5.0')
    time_limit_seconds: int = 3600
    trailing_stop: bool = False
    trailing_stop_activation_pct: Decimal = Decimal('2.0')
    trailing_stop_delta_pct: Decimal = Decimal('1.0')
    breakeven_trigger_pct: Decimal = Decimal('0')  # 0 = disabled; e.g. 1.5 = move SL to breakeven at +1.5%
    partial_tp: list[dict] = Field(default_factory=list)  # e.g. [{"pct": 3.0, "close_fraction": 0.5}]
    min_signal_confidence: float = 0.6
    exchange_fee_pct: float = 0.1  # per-side fee (0.4 Coinbase, 0.1 OKX, 0.045 HL)
    min_profit_after_fees_pct: float = 0.5  # min TP above round-trip fees (0 = disabled)
    correlation_groups: dict[str, list[str]] = Field(default_factory=dict)
    max_correlated_positions: int = 2
    volatility_circuit_breaker_multiplier: float = 0.0  # 0 = disabled
    volatility_lookback_candles: int = 20
    volatility_reference_pair: str = ''

    @field_validator(
        'stop_loss_pct',
        'take_profit_pct',
        'trailing_stop_activation_pct',
        'trailing_stop_delta_pct',
        'breakeven_trigger_pct',
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

    exchange: Literal['okx', 'bybit', 'coinbase', 'hyperliquid'] = 'okx'


class AppConfig(BaseModel):
    exchange: Literal['okx', 'bybit', 'coinbase', 'hyperliquid'] = 'okx'
    okx: OKXCredentials = Field(default_factory=OKXCredentials)
    bybit: BybitCredentials = Field(default_factory=BybitCredentials)
    coinbase: CoinbaseCredentials = Field(default_factory=CoinbaseCredentials)
    hyperliquid: HyperliquidCredentials = Field(default_factory=HyperliquidCredentials)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    market_selection: MarketSelectionConfig = Field(default_factory=MarketSelectionConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    strategy_params: dict = Field(default_factory=dict)


def validate_live_config(config: AppConfig) -> list[str]:
    """Validate that live-mode config has essential protections enabled.

    Returns a list of error messages. Empty list = all checks pass.
    Only called when RunningMode is LIVE_REAL_MONEY.
    """
    errors: list[str] = []
    risk = config.risk
    ms = config.market_selection
    trading = config.trading

    if risk.stop_loss_pct <= 0:
        errors.append('stop_loss_pct must be > 0 in live mode')
    if risk.take_profit_pct <= 0:
        errors.append('take_profit_pct must be > 0 in live mode')
    if risk.min_signal_confidence < 0.5:
        errors.append(
            f'min_signal_confidence={risk.min_signal_confidence} is too low for live '
            f'(minimum 0.5 required)'
        )
    if config.budget.daily_loss_limit_usdt <= 0:
        errors.append('daily_loss_limit_usdt must be > 0 in live mode')
    if risk.volatility_circuit_breaker_multiplier <= 0:
        errors.append(
            'volatility_circuit_breaker_multiplier must be > 0 in live mode '
            '(circuit breaker is required)'
        )
    if len(trading.pairs) >= 3 and not risk.correlation_groups:
        errors.append(
            f'correlation_groups must be configured when trading {len(trading.pairs)} '
            f'pairs in live mode'
        )
    if ms.min_volume_usdt <= 0 and ms.max_spread_pct <= 0:
        errors.append(
            'at least one market quality filter must be active in live mode '
            '(min_volume_usdt > 0 or max_spread_pct > 0)'
        )
    return errors


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
    hyperliquid = HyperliquidCredentials()
    telegram_toml = toml_data.get('telegram', {})
    telegram = TelegramConfig(**telegram_toml)
    exchange_selector = _ExchangeSelector()

    trading = TradingConfig(**toml_data.get('trading', {}))
    budget = BudgetConfig(**toml_data.get('budget', {}))
    risk = RiskConfig(**toml_data.get('risk', {}))
    market_selection = MarketSelectionConfig(**toml_data.get('market_selection', {}))

    strategy_params = toml_data.get('strategy', {}).get(trading.strategy, {})

    config = AppConfig(
        exchange=exchange_selector.exchange,
        okx=okx, bybit=bybit, coinbase=coinbase,
        hyperliquid=hyperliquid,
        trading=trading, budget=budget, risk=risk,
        market_selection=market_selection,
        telegram=telegram, strategy_params=strategy_params,
    )

    creds_map = {
        'okx': config.okx, 'bybit': config.bybit,
        'coinbase': config.coinbase, 'hyperliquid': config.hyperliquid,
    }
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
