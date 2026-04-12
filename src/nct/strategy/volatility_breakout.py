"""ATR-based volatility breakout strategy."""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.volatility

from nct.strategy.base import IStrategy


class VolatilityBreakoutStrategy(IStrategy):
    """Volatility breakout strategy using ATR and price range.

    BUY when:
      - Price breaks above the N-period high
      - Breakout distance exceeds ATR * multiplier (filters noise)

    SELL when:
      - Price breaks below the N-period low
      - Breakout distance exceeds ATR * multiplier

    Uses tighter stops than trend-following — breakout failure means
    quick exit. Performs best in low-volatility consolidation → expansion.
    """

    def __init__(
        self,
        *,
        lookback_period: int = 20,
        atr_period: int = 14,
        breakout_atr_multiplier: float = 0.5,
        atr_sl_multiplier: float = 1.0,
        atr_tp_multiplier: float = 2.0,
    ) -> None:
        self._lookback_period = lookback_period
        self._atr_period = atr_period
        self._breakout_atr_multiplier = breakout_atr_multiplier
        self._atr_sl_multiplier = atr_sl_multiplier
        self._atr_tp_multiplier = atr_tp_multiplier

    @property
    def name(self) -> str:
        return 'volatility_breakout'

    @property
    def required_candle_count(self) -> int:
        return max(self._lookback_period, self._atr_period) + 10

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        # N-period high/low (excluding current candle)
        dataframe['range_high'] = (
            dataframe['high'].shift(1).rolling(self._lookback_period).max()
        )
        dataframe['range_low'] = (
            dataframe['low'].shift(1).rolling(self._lookback_period).min()
        )
        dataframe['range_mid'] = (
            (dataframe['range_high'] + dataframe['range_low']) / 2
        )

        # ATR for breakout confirmation and SL/TP
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

        close = dataframe['close']
        atr = dataframe['atr']
        range_high = dataframe['range_high']
        range_low = dataframe['range_low']
        threshold = atr * self._breakout_atr_multiplier

        # Breakout above range high
        long_mask = (
            (close > range_high)
            & (close - range_high > threshold)
            & (close.shift(1) <= range_high.shift(1))
        )

        # Breakout below range low
        short_mask = (
            (close < range_low)
            & (range_low - close > threshold)
            & (close.shift(1) >= range_low.shift(1))
        )

        dataframe.loc[long_mask, 'enter_long'] = 1
        dataframe.loc[short_mask, 'enter_short'] = 1

        # Confidence: scaled by breakout magnitude relative to ATR
        for mask in (long_mask, short_mask):
            if mask.any():
                breakout_dist = abs(close[mask] - range_high[mask])
                conf = (breakout_dist / atr[mask]).clip(0.4, 1.0)
                dataframe.loc[mask, 'signal_confidence'] = conf

        # Reasons
        dataframe.loc[long_mask, 'signal_reason'] = (
            'Price broke above range high with ATR confirmation'
        )
        dataframe.loc[short_mask, 'signal_reason'] = (
            'Price broke below range low with ATR confirmation'
        )

        # ATR-based SL/TP (tighter stop for breakout failure)
        atr_pct = (atr / close) * 100
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

        close = dataframe['close']
        range_mid = dataframe['range_mid']

        # Exit long: price returns to range midpoint (breakout failed)
        dataframe.loc[
            (close < range_mid) & (close.shift(1) >= range_mid.shift(1)),
            'exit_long',
        ] = 1

        # Exit short: price returns to range midpoint
        dataframe.loc[
            (close > range_mid) & (close.shift(1) <= range_mid.shift(1)),
            'exit_short',
        ] = 1

        return dataframe
