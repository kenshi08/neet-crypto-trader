"""Bollinger Bands + volume mean reversion strategy."""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.volatility
import ta.volume

from nct.strategy.base import IStrategy


class MeanReversionStrategy(IStrategy):
    """Mean reversion strategy using Bollinger Bands and volume confirmation.

    BUY when:
      - Price touches or breaks below the lower Bollinger Band
      - Volume is above average (confirmation of genuine move, not noise)

    SELL when:
      - Price touches or breaks above the upper Bollinger Band

    Works best in ranging/sideways markets where momentum strategies fail.

    Confidence is scaled by how far price has deviated from the band
    and the volume multiplier.
    """

    def __init__(
        self,
        *,
        bb_period: int = 20,
        bb_std: float = 2.0,
        volume_ma_period: int = 20,
        volume_multiplier: float = 1.2,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 2.0,
    ) -> None:
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._volume_ma_period = volume_ma_period
        self._volume_multiplier = volume_multiplier
        self._atr_period = atr_period
        self._atr_sl_multiplier = atr_sl_multiplier
        self._atr_tp_multiplier = atr_tp_multiplier

    @property
    def name(self) -> str:
        return 'mean_reversion'

    @property
    def required_candle_count(self) -> int:
        return max(self._bb_period, self._volume_ma_period, self._atr_period) + 10

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            dataframe['close'],
            window=self._bb_period,
            window_dev=self._bb_std,
        )
        dataframe['bb_upper'] = bb.bollinger_hband()
        dataframe['bb_middle'] = bb.bollinger_mavg()
        dataframe['bb_lower'] = bb.bollinger_lband()
        dataframe['bb_width'] = bb.bollinger_wband()
        dataframe['bb_pct'] = bb.bollinger_pband()  # %B indicator

        # Volume moving average
        dataframe['volume_ma'] = dataframe['volume'].rolling(
            window=self._volume_ma_period,
        ).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']

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

        # Long: price at or below lower band + above-average volume
        long_mask = (
            (dataframe['close'] <= dataframe['bb_lower'])
            & (dataframe['volume_ratio'] >= self._volume_multiplier)
        )

        # Short: price at or above upper band + above-average volume
        short_mask = (
            (dataframe['close'] >= dataframe['bb_upper'])
            & (dataframe['volume_ratio'] >= self._volume_multiplier)
        )

        dataframe.loc[long_mask, 'enter_long'] = 1
        dataframe.loc[short_mask, 'enter_short'] = 1

        # Confidence: based on how far past the band + volume strength
        # %B < 0 means below lower band, more negative = more oversold
        dataframe.loc[long_mask, 'signal_confidence'] = (
            (1 - dataframe.loc[long_mask, 'bb_pct']).clip(0.3, 1.0)
            * (dataframe.loc[long_mask, 'volume_ratio'] / 3).clip(0.3, 1.0)
        )
        dataframe.loc[short_mask, 'signal_confidence'] = (
            dataframe.loc[short_mask, 'bb_pct'].clip(0.3, 1.0)
            * (dataframe.loc[short_mask, 'volume_ratio'] / 3).clip(0.3, 1.0)
        )

        dataframe.loc[long_mask, 'signal_reason'] = (
            'Price below lower Bollinger Band with above-average volume'
        )
        dataframe.loc[short_mask, 'signal_reason'] = (
            'Price above upper Bollinger Band with above-average volume'
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

        # Exit long: price reaches middle band (mean reversion target)
        # or breaks above upper band (overshoot)
        dataframe.loc[
            (dataframe['close'] >= dataframe['bb_middle']),
            'exit_long',
        ] = 1

        # Exit short: price reaches middle band or breaks below lower band
        dataframe.loc[
            (dataframe['close'] <= dataframe['bb_middle']),
            'exit_short',
        ] = 1

        return dataframe
