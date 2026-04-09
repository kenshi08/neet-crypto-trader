"""Exception hierarchy for neet-crypto-trader."""


class NCTError(Exception):
    """Base exception for neet-crypto-trader."""


class ConfigError(NCTError):
    """Invalid configuration."""


class ExchangeError(NCTError):
    """General exchange communication error."""


class RateLimitError(ExchangeError):
    """OKX rate limit exceeded (HTTP 429 or sCode indicates throttling)."""


class AuthenticationError(ExchangeError):
    """Invalid API credentials or signature."""


class OrderError(ExchangeError):
    """Order rejected by exchange."""


class StrategyError(NCTError):
    """Strategy computation failure."""


class BudgetExhaustedError(NCTError):
    """Budget limit reached — informational, not crash-worthy."""


class DataError(NCTError):
    """Missing or corrupt market data."""
