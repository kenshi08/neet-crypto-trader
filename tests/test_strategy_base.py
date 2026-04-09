"""Tests for the strategy base interface."""

from __future__ import annotations

import pandas as pd
import pytest

from nct.strategy.base import IStrategy, Signal, SignalResult, strategy_safe_wrapper


class _DummyStrategy(IStrategy):
    """Minimal strategy for testing the base class."""

    @property
    def name(self) -> str:
        return 'dummy'

    @property
    def required_candle_count(self) -> int:
        return 5

    def populate_indicators(self, dataframe, metadata):
        dataframe['indicator'] = dataframe['close'].rolling(3).mean()
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['signal_confidence'] = 0.0
        dataframe['signal_reason'] = ''
        # Buy when close > rolling mean
        mask = dataframe['close'] > dataframe['indicator']
        dataframe.loc[mask, 'enter_long'] = 1
        dataframe.loc[mask, 'signal_confidence'] = 0.75
        dataframe.loc[mask, 'signal_reason'] = 'Close above SMA'
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe


def _make_df(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='15min'),
        'open': [c - 0.5 for c in closes],
        'high': [c + 1.0 for c in closes],
        'low': [c - 1.0 for c in closes],
        'close': closes,
        'volume': [1000.0] * n,
    })


class TestSignalResult:
    def test_immutable(self):
        sr = SignalResult(signal=Signal.BUY, confidence=0.8, reason='test')
        with pytest.raises(AttributeError):
            sr.signal = Signal.SELL  # type: ignore[misc]

    def test_signal_enum(self):
        assert Signal.BUY == 'buy'
        assert Signal.SELL == 'sell'
        assert Signal.HOLD == 'hold'


class TestIStrategy:
    def test_evaluate_insufficient_candles(self):
        strategy = _DummyStrategy()
        df = _make_df([100.0, 101.0, 102.0])  # 3 < 5 required
        result = strategy.evaluate(df, {})
        assert result.signal == Signal.HOLD
        assert 'Insufficient candles' in result.reason

    def test_evaluate_produces_buy_signal(self):
        # Ascending prices → close > SMA → enter_long
        strategy = _DummyStrategy()
        df = _make_df([90, 92, 94, 96, 98, 100, 102])
        result = strategy.evaluate(df, {})
        assert result.signal == Signal.BUY
        assert result.confidence > 0

    def test_evaluate_produces_hold_on_flat(self):
        strategy = _DummyStrategy()
        # Flat prices → close == SMA → no signal (not >, just ==)
        df = _make_df([100, 100, 100, 100, 100, 100, 100])
        result = strategy.evaluate(df, {})
        assert result.signal == Signal.HOLD

    def test_does_not_modify_original_dataframe(self):
        strategy = _DummyStrategy()
        df = _make_df([90, 92, 94, 96, 98, 100, 102])
        original_cols = set(df.columns)
        strategy.evaluate(df, {})
        assert set(df.columns) == original_cols  # no mutation


class TestStrategySafeWrapper:
    def test_catches_exception(self):
        @strategy_safe_wrapper
        def bad_func(df):
            msg = 'boom'
            raise ValueError(msg)

        df = pd.DataFrame({'close': [1, 2, 3]})
        result = bad_func(df)
        # Returns the dataframe instead of crashing
        assert isinstance(result, pd.DataFrame)

    def test_passes_through_on_success(self):
        @strategy_safe_wrapper
        def good_func(df):
            df['new_col'] = 1
            return df

        df = pd.DataFrame({'close': [1, 2, 3]})
        result = good_func(df)
        assert 'new_col' in result.columns
