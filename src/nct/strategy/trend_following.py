"""EMA crossover + ADX trend-following strategy."""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.trend
import ta.volatility

from nct.strategy.base import IStrategy


class TrendFollowingStrategy(IStrategy):
    """Trend-following strategy combining EMA crossover with ADX filter.

    BUY when:
      - Fast EMA crosses above slow EMA (golden cross)
      - ADX > threshold (confirming trend strength, not range-bound)

    SELL when:
      - Fast EMA crosses below slow EMA (death cross)
      - ADX > threshold

    Orthogonal to momentum (RSI/MACD) and mean-reversion (Bollinger).
    Performs best in sustained directional moves, struggles in chop.
    """

    def __init__(
        self,
        *,
        ema_fast: int = 9,
        ema_slow: int = 21,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
        atr_sl_multiplier: float = 2.0,
        atr_tp_multiplier: float = 3.0,
    ) -> None:
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow
        self._adx_period = adx_period
        self._adx_threshold = adx_threshold
        self._atr_period = atr_period
        self._atr_sl_multiplier = atr_sl_multiplier
        self._atr_tp_multiplier = atr_tp_multiplier

    @property
    def name(self) -> str:
        return 'trend_following'

    @property
    def required_candle_count(self) -> int:
        return max(self._ema_slow, self._adx_period, self._atr_period) + 10

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        # EMAs
        ema_fast = ta.trend.EMAIndicator(
            dataframe['close'], window=self._ema_fast,
        )
        ema_slow = ta.trend.EMAIndicator(
            dataframe['close'], window=self._ema_slow,
        )
        dataframe['ema_fast'] = ema_fast.ema_indicator()
        dataframe['ema_slow'] = ema_slow.ema_indicator()

        # ADX — trend strength indicator (0-100, >25 = trending)
        adx = ta.trend.ADXIndicator(
            dataframe['high'], dataframe['low'], dataframe['close'],
            window=self._adx_period,
        )
        dataframe['adx'] = adx.adx()

        # ATR for dynamic SL/TP
        atr = ta.volatility.AverageTrueRange(
            dataframe['high'], dataframe['low'], dataframe['close'],
            window=self._atr_period,
        )
        dataframe['atr'] = atr.average_true_range()

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['signal_confidence'] = 0.0
        dataframe['signal_reason'] = ''
        dataframe['suggested_sl_pct'] = None
        dataframe['suggested_tp_pct'] = None

        ema_fast = dataframe['ema_fast']
        ema_slow = dataframe['ema_slow']
        adx = dataframe['adx']

        # Golden cross: fast EMA crosses above slow EMA + strong trend
        long_mask = (
            (ema_fast > ema_slow)
            & (ema_fast.shift(1) <= ema_slow.shift(1))
            & (adx > self._adx_threshold)
        )

        # Death cross: fast EMA crosses below slow EMA + strong trend
        short_mask = (
            (ema_fast < ema_slow)
            & (ema_fast.shift(1) >= ema_slow.shift(1))
            & (adx > self._adx_threshold)
        )

        dataframe.loc[long_mask, 'enter_long'] = 1
        dataframe.loc[short_mask, 'enter_short'] = 1

        # Confidence: scaled by ADX strength (25-50 → 0.5-1.0)
        dataframe.loc[long_mask, 'signal_confidence'] = (
            (adx[long_mask] - self._adx_threshold)
            / (50 - self._adx_threshold)
        ).clip(0.4, 1.0)

        dataframe.loc[short_mask, 'signal_confidence'] = (
            (adx[short_mask] - self._adx_threshold)
            / (50 - self._adx_threshold)
        ).clip(0.4, 1.0)

        # Reasons
        dataframe.loc[long_mask, 'signal_reason'] = (
            'EMA golden cross + ADX confirms trend'
        )
        dataframe.loc[short_mask, 'signal_reason'] = (
            'EMA death cross + ADX confirms trend'
        )

        # ATR-based SL/TP
        atr_pct = (dataframe['atr'] / dataframe['close']) * 100
        entry_mask = long_mask | short_mask
        dataframe.loc[entry_mask, 'suggested_sl_pct'] = (
            atr_pct[entry_mask] * self._atr_sl_multiplier
        )
        dataframe.loc[entry_mask, 'suggested_tp_pct'] = (
            atr_pct[entry_mask] * self._atr_tp_multiplier
        )

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        ema_fast = dataframe['ema_fast']
        ema_slow = dataframe['ema_slow']
        adx = dataframe['adx']

        # Exit long: fast EMA crosses below slow OR ADX collapses
        dataframe.loc[
            (
                (ema_fast < ema_slow)
                & (ema_fast.shift(1) >= ema_slow.shift(1))
            )
            | (adx < self._adx_threshold * 0.6),
            'exit_long',
        ] = 1

        # Exit short: fast EMA crosses above slow OR ADX collapses
        dataframe.loc[
            (
                (ema_fast > ema_slow)
                & (ema_fast.shift(1) <= ema_slow.shift(1))
            )
            | (adx < self._adx_threshold * 0.6),
            'exit_short',
        ] = 1

        return dataframe
