"""ATR-based volatility breakout strategy."""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.trend
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
        volume_ma_period: int = 20,
        volume_multiplier: float = 1.2,
        keltner_period: int = 20,
        keltner_atr_multiplier: float = 1.5,
        use_squeeze: bool = False,
    ) -> None:
        self._lookback_period = lookback_period
        self._atr_period = atr_period
        self._breakout_atr_multiplier = breakout_atr_multiplier
        self._atr_sl_multiplier = atr_sl_multiplier
        self._atr_tp_multiplier = atr_tp_multiplier
        self._volume_ma_period = volume_ma_period
        self._volume_multiplier = volume_multiplier
        self._keltner_period = keltner_period
        self._keltner_atr_multiplier = keltner_atr_multiplier
        self._use_squeeze = use_squeeze

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

        # Volume confirmation (#96)
        dataframe['volume_ma'] = dataframe['volume'].rolling(
            window=self._volume_ma_period,
        ).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']

        # TTM Squeeze detection (#98)
        if self._use_squeeze:
            bb = ta.volatility.BollingerBands(
                dataframe['close'],
                window=self._keltner_period,
                window_dev=2,
            )
            dataframe['bb_upper'] = bb.bollinger_hband()
            dataframe['bb_lower'] = bb.bollinger_lband()

            kc_atr = ta.volatility.AverageTrueRange(
                dataframe['high'], dataframe['low'],
                dataframe['close'], window=self._keltner_period,
            ).average_true_range()
            kc_ema = ta.trend.EMAIndicator(
                dataframe['close'], window=self._keltner_period,
            ).ema_indicator()
            dataframe['kc_upper'] = kc_ema + kc_atr * self._keltner_atr_multiplier
            dataframe['kc_lower'] = kc_ema - kc_atr * self._keltner_atr_multiplier

            # Squeeze: BB inside KC = compression
            squeeze_on = (
                (dataframe['bb_lower'] > dataframe['kc_lower'])
                & (dataframe['bb_upper'] < dataframe['kc_upper'])
            )
            squeeze_off = ~squeeze_on
            # Squeeze just fired: was on, now off (within last 3 bars)
            dataframe['squeeze_fired'] = (
                squeeze_off & squeeze_on.shift(1).rolling(3).max().fillna(0).astype(bool)
            )
        else:
            dataframe['squeeze_fired'] = True  # no filter when disabled

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
        vol_ok = dataframe['volume_ratio'] >= self._volume_multiplier
        squeeze_ok = dataframe['squeeze_fired']

        # Breakout above range high + volume + squeeze
        long_mask = (
            (close > range_high)
            & (close - range_high > threshold)
            & (close.shift(1) <= range_high.shift(1))
            & vol_ok
            & squeeze_ok
        )

        # Breakout below range low + volume + squeeze
        short_mask = (
            (close < range_low)
            & (range_low - close > threshold)
            & (close.shift(1) >= range_low.shift(1))
            & vol_ok
            & squeeze_ok
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
