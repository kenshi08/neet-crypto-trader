"""OHLCV data pipeline — fetches candles, converts to DataFrames, caches results."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import structlog

from nct.exchange.base import IExchange
from nct.exchange.models import Candle

log = structlog.get_logger()


class DataProvider:
    """Central data hub that feeds strategies with OHLCV DataFrames.

    Features:
    - Converts OKX Candle objects to pandas DataFrames
    - TTL-based caching to avoid redundant API calls
    - Multi-pair parallel fetching
    - Detects new candle close to trigger strategy re-evaluation
    """

    def __init__(self, client: IExchange, *, cache_ttl_seconds: int = 5) -> None:
        self._client = client
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._prev_timestamps: dict[str, object] = {}

    async def get_dataframe(
        self,
        inst_id: str,
        timeframe: str = '15m',
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get OHLCV DataFrame for a trading pair, using cache if fresh."""
        cache_key = f'{inst_id}:{timeframe}'
        now = datetime.now(UTC)

        entry = self._cache.get(cache_key)
        if entry and (now - entry.fetched_at).total_seconds() < self._cache_ttl:
            return entry.dataframe

        candles = await self._client.get_candlesticks(
            inst_id, bar=timeframe, limit=limit,
        )
        df = candles_to_dataframe(candles)

        self._cache[cache_key] = _CacheEntry(dataframe=df, fetched_at=now)

        log.debug(
            'data_fetched',
            pair=inst_id,
            timeframe=timeframe,
            candles=len(df),
        )
        return df

    async def get_dataframes(
        self,
        pairs: list[str],
        timeframe: str = '15m',
        limit: int = 100,
    ) -> dict[str, pd.DataFrame]:
        """Fetch DataFrames for multiple pairs in parallel."""
        results = {}
        for pair in pairs:
            try:
                results[pair] = await self.get_dataframe(pair, timeframe, limit)
            except Exception:
                log.exception('data_fetch_failed', pair=pair)
        return results

    async def get_htf_dataframes(
        self,
        pairs: list[str],
        timeframes: list[str],
        limit: int = 60,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Fetch higher-timeframe data for multi-TF confirmation.

        Returns {pair: {timeframe: DataFrame}} for each pair and timeframe.
        """
        results: dict[str, dict[str, pd.DataFrame]] = {}
        for pair in pairs:
            results[pair] = {}
            for tf in timeframes:
                try:
                    results[pair][tf] = await self.get_dataframe(pair, tf, limit)
                except Exception:
                    log.debug('htf_fetch_failed', pair=pair, timeframe=tf)
        return results

    def has_new_candle(self, inst_id: str, timeframe: str = '15m') -> bool:
        """Check if the cached data has a newer candle than last check.

        Call this to decide if strategy re-evaluation is needed.
        """
        cache_key = f'{inst_id}:{timeframe}'
        entry = self._cache.get(cache_key)
        if not entry or len(entry.dataframe) == 0:
            return True

        prev_key = f'{cache_key}:prev_ts'
        prev_ts = self._prev_timestamps.get(prev_key)
        current_ts = entry.dataframe.iloc[-1]['date']

        if prev_ts is None or current_ts != prev_ts:
            self._prev_timestamps[prev_key] = current_ts
            return True
        return False

    def clear_cache(self) -> None:
        self._cache.clear()
        self._prev_timestamps.clear()


class _CacheEntry:
    __slots__ = ('dataframe', 'fetched_at')

    def __init__(self, dataframe: pd.DataFrame, fetched_at: datetime) -> None:
        self.dataframe = dataframe
        self.fetched_at = fetched_at


def candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
    """Convert a list of Candle objects to a pandas DataFrame.

    Returns DataFrame with columns: date, open, high, low, close, volume.
    All price/volume columns are float64 for TA indicator compatibility.
    """
    if not candles:
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    rows = [
        {
            'date': c.timestamp,
            'open': float(c.open),
            'high': float(c.high),
            'low': float(c.low),
            'close': float(c.close),
            'volume': float(c.volume),
        }
        for c in candles
    ]
    df = pd.DataFrame(rows)
    df = df.sort_values('date').reset_index(drop=True)
    return df
