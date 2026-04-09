"""RSI + MACD momentum strategy for short-term speculation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.momentum
import ta.trend
import ta.volatility

from nct.strategy.base import IStrategy


class MomentumStrategy(IStrategy):
    """Momentum strategy combining RSI and MACD for entry/exit signals.

    BUY when:
      - RSI crosses below oversold threshold (potential reversal up)
      - MACD histogram turns positive (momentum shifting bullish)

    SELL when:
      - RSI crosses above overbought threshold (potential reversal down)
      - MACD histogram turns negative (momentum shifting bearish)

    Confidence is scaled by RSI extremity and MACD histogram magnitude.
    Stop-loss/take-profit are derived from ATR for volatility-adaptive exits.
    """

    def __init__(
        self,
        *,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 2.5,
    ) -> None:
        self._rsi_period = rsi_period
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._atr_period = atr_period
        self._atr_sl_multiplier = atr_sl_multiplier
        self._atr_tp_multiplier = atr_tp_multiplier

    @property
    def name(self) -> str:
        return 'momentum'

    @property
    def required_candle_count(self) -> int:
        return max(self._macd_slow, self._rsi_period, self._atr_period) + 10

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        # RSI
        rsi = ta.momentum.RSIIndicator(dataframe['close'], window=self._rsi_period)
        dataframe['rsi'] = rsi.rsi()

        # MACD
        macd = ta.trend.MACD(
            dataframe['close'],
            window_fast=self._macd_fast,
            window_slow=self._macd_slow,
            window_sign=self._macd_signal,
        )
        dataframe['macd'] = macd.macd()
        dataframe['macd_signal'] = macd.macd_signal()
        dataframe['macd_hist'] = macd.macd_diff()

        # ATR for dynamic SL/TP
        atr = ta.volatility.AverageTrueRange(
            dataframe['high'], dataframe['low'], dataframe['close'],
            window=self._atr_period,
        )
        dataframe['atr'] = atr.average_true_range()

        # EMA for trend filter
        ema_fast = ta.trend.EMAIndicator(dataframe['close'], window=9)
        ema_slow = ta.trend.EMAIndicator(dataframe['close'], window=21)
        dataframe['ema_fast'] = ema_fast.ema_indicator()
        dataframe['ema_slow'] = ema_slow.ema_indicator()

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

        # Long entry: RSI oversold + MACD histogram crossing positive
        long_mask = (
            (dataframe['rsi'] < self._rsi_oversold)
            & (dataframe['macd_hist'] > 0)
            & (dataframe['macd_hist'].shift(1) <= 0)  # histogram just turned positive
        )

        # Short entry: RSI overbought + MACD histogram crossing negative
        short_mask = (
            (dataframe['rsi'] > self._rsi_overbought)
            & (dataframe['macd_hist'] < 0)
            & (dataframe['macd_hist'].shift(1) >= 0)  # histogram just turned negative
        )

        dataframe.loc[long_mask, 'enter_long'] = 1
        dataframe.loc[short_mask, 'enter_short'] = 1

        # Confidence: based on RSI extremity
        dataframe.loc[long_mask, 'signal_confidence'] = (
            (self._rsi_oversold - dataframe.loc[long_mask, 'rsi']) / self._rsi_oversold
        ).clip(0.3, 1.0)

        dataframe.loc[short_mask, 'signal_confidence'] = (
            (dataframe.loc[short_mask, 'rsi'] - self._rsi_overbought)
            / (100 - self._rsi_overbought)
        ).clip(0.3, 1.0)

        # Reasons
        dataframe.loc[long_mask, 'signal_reason'] = (
            'RSI oversold + MACD histogram turned positive'
        )
        dataframe.loc[short_mask, 'signal_reason'] = (
            'RSI overbought + MACD histogram turned negative'
        )

        # ATR-based SL/TP (as percentage of price)
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

        # Exit long: RSI overbought OR MACD histogram crosses negative
        dataframe.loc[
            (dataframe['rsi'] > self._rsi_overbought)
            | (
                (dataframe['macd_hist'] < 0)
                & (dataframe['macd_hist'].shift(1) >= 0)
            ),
            'exit_long',
        ] = 1

        # Exit short: RSI oversold OR MACD histogram crosses positive
        dataframe.loc[
            (dataframe['rsi'] < self._rsi_oversold)
            | (
                (dataframe['macd_hist'] > 0)
                & (dataframe['macd_hist'].shift(1) <= 0)
            ),
            'exit_short',
        ] = 1

        return dataframe
