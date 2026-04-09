"""Tests for the OHLCV data pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd

from nct.exchange.models import Candle
from nct.strategy.data_provider import DataProvider, candles_to_dataframe


def _sample_candles(n: int = 10) -> list[Candle]:
    from datetime import timedelta

    base = 67000
    start = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
    return [
        Candle(
            timestamp=start + timedelta(minutes=i * 15),
            open=Decimal(str(base + i * 10)),
            high=Decimal(str(base + i * 10 + 50)),
            low=Decimal(str(base + i * 10 - 30)),
            close=Decimal(str(base + i * 10 + 20)),
            volume=Decimal(str(1000 + i * 100)),
        )
        for i in range(n)
    ]


class TestCandlesToDataframe:
    def test_converts_candles(self):
        candles = _sample_candles(5)
        df = candles_to_dataframe(candles)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['date', 'open', 'high', 'low', 'close', 'volume']
        assert len(df) == 5

    def test_columns_are_float(self):
        df = candles_to_dataframe(_sample_candles(3))
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert df[col].dtype == float

    def test_sorted_by_date(self):
        candles = _sample_candles(5)
        # Reverse to test sorting
        df = candles_to_dataframe(list(reversed(candles)))
        dates = df['date'].tolist()
        assert dates == sorted(dates)

    def test_empty_candles(self):
        df = candles_to_dataframe([])
        assert len(df) == 0
        assert list(df.columns) == ['date', 'open', 'high', 'low', 'close', 'volume']

    def test_values_correct(self):
        candles = [
            Candle(
                timestamp=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
                open=Decimal('67000'),
                high=Decimal('67500'),
                low=Decimal('66800'),
                close=Decimal('67200'),
                volume=Decimal('1234.56'),
            ),
        ]
        df = candles_to_dataframe(candles)
        row = df.iloc[0]
        assert row['open'] == 67000.0
        assert row['high'] == 67500.0
        assert row['low'] == 66800.0
        assert row['close'] == 67200.0
        assert row['volume'] == 1234.56


class TestDataProviderCache:
    def test_has_new_candle_on_first_call(self):
        # DataProvider needs an OKXClient, but has_new_candle only checks cache
        # We test the cache logic without making API calls
        from unittest.mock import MagicMock

        provider = DataProvider(MagicMock(), cache_ttl_seconds=5)
        # No cache = always new
        assert provider.has_new_candle('BTC-USDT') is True

    def test_clear_cache(self):
        from unittest.mock import MagicMock

        provider = DataProvider(MagicMock(), cache_ttl_seconds=5)
        provider._prev_timestamps['test'] = 'value'
        provider.clear_cache()
        assert len(provider._prev_timestamps) == 0
        assert len(provider._cache) == 0
