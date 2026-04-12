"""Macro factor layer -- structural economic data that shifts crypto regimes.

Provides hard economic data (not rumors): Fed rates, CPI, DXY, VIX,
S&P 500, Fear & Greed Index, and economic calendar event proximity.
All data sources are free.  Each source degrades gracefully if unavailable.

Data refresh cadence:
  - DXY / S&P 500 / VIX: every 5 minutes (intraday)
  - Fear & Greed: every hour (daily index)
  - FRED (rates, CPI): every 6 hours (changes monthly)
  - Economic calendar: every 12 hours (static schedule)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MacroFeatures:
    """Quantitative macro factor snapshot -- all fields are optional."""

    # Rate environment (FRED)
    fed_funds_rate: float | None = None
    treasury_10y_yield: float | None = None

    # Inflation (FRED)
    cpi_yoy: float | None = None        # Year-over-year CPI %
    cpi_surprise: float | None = None    # Actual - Expected (if available)

    # Dollar strength (yfinance)
    dxy_value: float | None = None       # Current DXY level
    dxy_zscore: float | None = None      # 20-day z-score
    dxy_roc_5d: float | None = None      # 5-day rate of change %

    # Risk appetite (yfinance)
    sp500_roc_1d: float | None = None    # S&P 500 1-day return %
    vix_level: float | None = None       # Current VIX
    vix_zscore: float | None = None      # 20-day z-score

    # Sentiment (alternative.me)
    fear_greed: int | None = None        # 0-100 index
    fear_greed_label: str | None = None  # e.g. 'Extreme Fear', 'Greed'

    # Event proximity (economic calendar)
    hours_to_next_high_impact: float | None = None
    hours_since_last_high_impact: float | None = None
    next_event_name: str | None = None
    is_event_window: bool = False        # Within +/-4 hours of high-impact event

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class EconomicEvent:
    """A scheduled macro event from the economic calendar."""

    name: str
    dt_utc: datetime
    impact: str  # 'high', 'medium', 'low'
    currency: str  # 'USD', etc.


# ---------------------------------------------------------------------------
# TTL cache helper
# ---------------------------------------------------------------------------

class _TTLCache:
    """Simple in-memory cache with per-key TTL in seconds."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int) -> None:
        self._store[key] = (value, time.monotonic() + ttl)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Individual data source fetchers
# ---------------------------------------------------------------------------

async def fetch_fear_greed() -> tuple[int, str] | None:
    """Fetch the Crypto Fear & Greed Index from alternative.me (free, no key).

    Returns (score, label) or None on failure.
    """
    import json
    import urllib.request

    url = 'https://api.alternative.me/fng/?limit=1'
    try:
        loop = asyncio.get_running_loop()
        response_text = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(url, timeout=10).read().decode(),
        )
        data = json.loads(response_text)
        entry = data['data'][0]
        return int(entry['value']), entry['value_classification']
    except Exception:
        log.warning('fear_greed_fetch_failed', exc_info=True)
        return None


async def fetch_yfinance_snapshot(
    symbols: list[str],
) -> dict[str, dict[str, float]]:
    """Fetch current price + 20-day stats for symbols via yfinance.

    Returns {symbol: {'close': float, 'zscore_20d': float, 'roc_5d': float, ...}}
    """
    try:
        import yfinance as yf

        loop = asyncio.get_running_loop()

        def _download() -> dict[str, dict[str, float]]:
            result: dict[str, dict[str, float]] = {}
            for sym in symbols:
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(period='1mo', interval='1d')
                    if hist.empty or len(hist) < 5:
                        continue
                    close = hist['Close']
                    current = float(close.iloc[-1])
                    mean_20 = float(close.tail(20).mean())
                    std_20 = float(close.tail(20).std())
                    zscore = (current - mean_20) / std_20 if std_20 > 0 else 0.0
                    roc_5d = (
                        (current - float(close.iloc[-6])) / float(close.iloc[-6]) * 100
                        if len(close) >= 6
                        else 0.0
                    )
                    roc_1d = (
                        (current - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
                        if len(close) >= 2
                        else 0.0
                    )
                    result[sym] = {
                        'close': current,
                        'zscore_20d': zscore,
                        'roc_5d': roc_5d,
                        'roc_1d': roc_1d,
                    }
                except Exception:
                    log.warning('yfinance_symbol_failed', symbol=sym, exc_info=True)
            return result

        return await loop.run_in_executor(None, _download)
    except ImportError:
        log.warning('yfinance_not_installed')
        return {}
    except Exception:
        log.warning('yfinance_fetch_failed', exc_info=True)
        return {}


async def fetch_fred_series(
    series_ids: list[str], *, api_key: str | None = None,
) -> dict[str, float]:
    """Fetch latest values from FRED API.

    If api_key is None, tries FRED_API_KEY env var. Returns {series_id: latest_value}.
    """
    if api_key is None:
        import os
        api_key = os.environ.get('FRED_API_KEY')

    if not api_key:
        log.debug('fred_no_api_key')
        return {}

    try:
        from fredapi import Fred

        loop = asyncio.get_running_loop()

        def _fetch() -> dict[str, float]:
            fred = Fred(api_key=api_key)
            result: dict[str, float] = {}
            for sid in series_ids:
                try:
                    s = fred.get_series(sid)
                    if s is not None and len(s) > 0:
                        result[sid] = float(s.dropna().iloc[-1])
                except Exception:
                    log.warning('fred_series_failed', series=sid, exc_info=True)
            return result

        return await loop.run_in_executor(None, _fetch)
    except ImportError:
        log.warning('fredapi_not_installed')
        return {}
    except Exception:
        log.warning('fred_fetch_failed', exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Economic calendar (built-in schedule of known recurring events)
# ---------------------------------------------------------------------------

# Hard-coded schedule of major US macro events.
# In production, this could be extended with a live calendar API.
_RECURRING_EVENTS = [
    # FOMC meetings 2026 (scheduled dates — 8 per year)
    # These are approximate; update annually.
    {'name': 'FOMC Rate Decision', 'impact': 'high', 'currency': 'USD'},
    {'name': 'CPI Release', 'impact': 'high', 'currency': 'USD'},
    {'name': 'Non-Farm Payrolls', 'impact': 'high', 'currency': 'USD'},
    {'name': 'PPI Release', 'impact': 'medium', 'currency': 'USD'},
    {'name': 'GDP Release', 'impact': 'medium', 'currency': 'USD'},
]


def get_upcoming_events(
    events: list[EconomicEvent] | None = None,
) -> tuple[EconomicEvent | None, EconomicEvent | None]:
    """Return (next_event, last_event) relative to now.

    If events is None, returns (None, None) — the caller should provide
    events from an external calendar source or pre-loaded schedule.
    """
    if not events:
        return None, None

    now = datetime.now(UTC)
    future = [e for e in events if e.dt_utc > now]
    past = [e for e in events if e.dt_utc <= now]

    next_event = min(future, key=lambda e: e.dt_utc) if future else None
    last_event = max(past, key=lambda e: e.dt_utc) if past else None

    return next_event, last_event


# ---------------------------------------------------------------------------
# Main provider
# ---------------------------------------------------------------------------

class MacroDataProvider:
    """Async provider for macro factor data with TTL caching.

    Each data source is independently optional and degrades gracefully.
    """

    # Cache TTLs in seconds
    _TTL_YFINANCE = 300       # 5 minutes
    _TTL_FEAR_GREED = 3600    # 1 hour
    _TTL_FRED = 21600         # 6 hours
    _TTL_CALENDAR = 43200     # 12 hours

    def __init__(
        self,
        *,
        fred_api_key: str | None = None,
        event_window_hours: float = 4.0,
        economic_events: list[EconomicEvent] | None = None,
    ) -> None:
        self._fred_key = fred_api_key
        self._event_window = event_window_hours
        self._events = economic_events or []
        self._cache = _TTLCache()

    async def get_macro_features(self) -> MacroFeatures:
        """Fetch all available macro data, returning whatever is available."""
        # Launch independent fetches concurrently
        fg_task = self._cached_fear_greed()
        yf_task = self._cached_yfinance()
        fred_task = self._cached_fred()

        fg_result, yf_result, fred_result = await asyncio.gather(
            fg_task, yf_task, fred_task, return_exceptions=True,
        )

        # Handle exceptions gracefully
        if isinstance(fg_result, Exception):
            log.warning('macro_fear_greed_error', error=str(fg_result))
            fg_result = None
        if isinstance(yf_result, Exception):
            log.warning('macro_yfinance_error', error=str(yf_result))
            yf_result = {}
        if isinstance(fred_result, Exception):
            log.warning('macro_fred_error', error=str(fred_result))
            fred_result = {}

        # Extract yfinance data
        dxy = yf_result.get('DX-Y.NYB', {}) if isinstance(yf_result, dict) else {}
        spx = yf_result.get('^GSPC', {}) if isinstance(yf_result, dict) else {}
        vix = yf_result.get('^VIX', {}) if isinstance(yf_result, dict) else {}

        # Extract FRED data
        fred = fred_result if isinstance(fred_result, dict) else {}

        # Fear & Greed
        fear_greed_val = None
        fear_greed_label = None
        if fg_result is not None:
            fear_greed_val, fear_greed_label = fg_result

        # Economic calendar
        next_event, last_event = get_upcoming_events(self._events)
        now = datetime.now(UTC)

        hours_to_next = None
        next_event_name = None
        if next_event:
            hours_to_next = (next_event.dt_utc - now).total_seconds() / 3600
            next_event_name = next_event.name

        hours_since_last = None
        if last_event:
            hours_since_last = (now - last_event.dt_utc).total_seconds() / 3600

        is_event_window = False
        if hours_to_next is not None and hours_to_next <= self._event_window:
            is_event_window = True
        if hours_since_last is not None and hours_since_last <= self._event_window:
            is_event_window = True

        return MacroFeatures(
            fed_funds_rate=fred.get('FEDFUNDS'),
            treasury_10y_yield=fred.get('DGS10'),
            cpi_yoy=fred.get('CPIAUCSL'),
            cpi_surprise=None,  # Requires expected value from external source
            dxy_value=dxy.get('close'),
            dxy_zscore=dxy.get('zscore_20d'),
            dxy_roc_5d=dxy.get('roc_5d'),
            sp500_roc_1d=spx.get('roc_1d'),
            vix_level=vix.get('close'),
            vix_zscore=vix.get('zscore_20d'),
            fear_greed=fear_greed_val,
            fear_greed_label=fear_greed_label,
            hours_to_next_high_impact=hours_to_next,
            hours_since_last_high_impact=hours_since_last,
            next_event_name=next_event_name,
            is_event_window=is_event_window,
        )

    # ------------------------------------------------------------------
    # Cached fetchers
    # ------------------------------------------------------------------

    async def _cached_fear_greed(self) -> tuple[int, str] | None:
        cached = self._cache.get('fear_greed')
        if cached is not None:
            return cached
        result = await fetch_fear_greed()
        if result is not None:
            self._cache.set('fear_greed', result, self._TTL_FEAR_GREED)
        return result

    async def _cached_yfinance(self) -> dict[str, dict[str, float]]:
        cached = self._cache.get('yfinance')
        if cached is not None:
            return cached
        result = await fetch_yfinance_snapshot(['DX-Y.NYB', '^GSPC', '^VIX'])
        # Cache even empty results to avoid hammering the API
        self._cache.set('yfinance', result, self._TTL_YFINANCE)
        return result

    async def _cached_fred(self) -> dict[str, float]:
        cached = self._cache.get('fred')
        if cached is not None:
            return cached
        result = await fetch_fred_series(
            ['FEDFUNDS', 'DGS10', 'CPIAUCSL'],
            api_key=self._fred_key,
        )
        self._cache.set('fred', result, self._TTL_FRED)
        return result


# ---------------------------------------------------------------------------
# Trading rules (standalone, usable before meta-model integration)
# ---------------------------------------------------------------------------

def macro_position_scale(features: MacroFeatures) -> float:
    """Return a position size multiplier [0.25, 1.0] based on macro conditions.

    This provides immediate value before the XGBoost meta-model is built.
    Rules are quantitative, not speculative.
    """
    scale = 1.0

    # Event window guard: reduce by 50% near FOMC/CPI/NFP
    if features.is_event_window:
        scale *= 0.5

    # VIX regime: reduce when equity fear is elevated
    if features.vix_level is not None and features.vix_level > 30:
        scale *= 0.7

    # DXY divergence: strong dollar = risk-off for crypto
    if features.dxy_zscore is not None and features.dxy_zscore > 2.0:
        scale *= 0.8

    # Extreme fear/greed (contrarian — don't reduce, but flag)
    # This doesn't reduce size; the meta-model will use it directionally.

    return max(0.25, scale)


def macro_directional_bias(features: MacroFeatures) -> float:
    """Return a directional bias [-1.0, 1.0] from macro factors.

    Positive = bullish for crypto, negative = bearish.
    Used as a feature in the meta-model, not as a standalone signal.
    """
    bias = 0.0

    # DXY: inverse correlation with crypto
    if features.dxy_roc_5d is not None:
        if features.dxy_roc_5d > 1.0:
            bias -= 0.3  # Dollar strengthening = bearish crypto
        elif features.dxy_roc_5d < -1.0:
            bias += 0.3  # Dollar weakening = bullish crypto

    # S&P 500: positive correlation (risk-on/risk-off)
    if features.sp500_roc_1d is not None:
        if features.sp500_roc_1d > 1.0:
            bias += 0.2
        elif features.sp500_roc_1d < -1.0:
            bias -= 0.2

    # Fear & Greed: contrarian
    if features.fear_greed is not None:
        if features.fear_greed < 15:
            bias += 0.3  # Extreme fear = contrarian bullish
        elif features.fear_greed > 85:
            bias -= 0.3  # Extreme greed = contrarian bearish

    # VIX: inverse (high VIX = fear = bearish)
    if features.vix_zscore is not None and features.vix_zscore > 2.0:
        bias -= 0.2

    return max(-1.0, min(1.0, bias))
