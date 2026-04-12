"""Base strategy interface and signal types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger()


class Signal(StrEnum):
    BUY = 'buy'
    SELL = 'sell'
    HOLD = 'hold'


@dataclass(frozen=True, slots=True)
class SignalResult:
    signal: Signal
    confidence: float
    reason: str
    suggested_stop_loss_pct: float | None = None
    suggested_take_profit_pct: float | None = None


class IStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses implement three methods that operate on pandas DataFrames:
    - populate_indicators: add TA columns
    - populate_entry_trend: set enter_long/enter_short columns
    - populate_exit_trend: set exit_long/exit_short columns

    The same code runs in both backtesting and live trading.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and config reference."""
        ...

    @property
    @abstractmethod
    def required_candle_count(self) -> int:
        """Minimum candles needed for indicator calculation."""
        ...

    @abstractmethod
    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Add technical indicator columns to the OHLCV dataframe."""
        ...

    @abstractmethod
    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Set entry signal columns: 'enter_long' and/or 'enter_short'."""
        ...

    @abstractmethod
    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Set exit signal columns: 'exit_long' and/or 'exit_short'."""
        ...

    def evaluate(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
        htf_data: dict[str, pd.DataFrame] | None = None,
    ) -> SignalResult:
        """Run the full strategy pipeline and return a signal for the latest candle.

        This is the main entry point called by the trading loop.

        Args:
            htf_data: Optional higher-timeframe DataFrames keyed by timeframe
                      (e.g. {'1H': df_1h, '4H': df_4h}). Used for multi-TF
                      confirmation — counter-trend signals get reduced confidence.
        """
        if len(dataframe) < self.required_candle_count:
            return SignalResult(
                signal=Signal.HOLD,
                confidence=0.0,
                reason=(
                    f'Insufficient candles: {len(dataframe)} '
                    f'< {self.required_candle_count} required'
                ),
            )

        df = dataframe.copy()
        df = self.populate_indicators(df, metadata)
        df = self.populate_entry_trend(df, metadata)
        df = self.populate_exit_trend(df, metadata)

        result = self._extract_signal(df, metadata)

        # Multi-timeframe confirmation (#97): reduce confidence on
        # signals that disagree with higher-timeframe trend direction.
        if htf_data and result.signal != Signal.HOLD:
            result = self._apply_htf_filter(result, htf_data)

        return result

    def _apply_htf_filter(
        self, result: SignalResult, htf_data: dict[str, pd.DataFrame],
    ) -> SignalResult:
        """Reduce confidence if the signal contradicts the higher-TF trend."""
        for _tf, htf_df in htf_data.items():
            if len(htf_df) < 50:
                continue
            # Simple HTF trend: close above/below EMA50
            ema50 = htf_df['close'].ewm(span=50, adjust=False).mean()
            htf_close = htf_df['close'].iloc[-1]
            htf_ema = ema50.iloc[-1]

            htf_bullish = htf_close > htf_ema
            signal_bullish = result.signal == Signal.BUY

            if htf_bullish != signal_bullish:
                # Counter-trend: halve confidence
                return SignalResult(
                    signal=result.signal,
                    confidence=result.confidence * 0.5,
                    reason=result.reason + ' (counter-HTF, confidence halved)',
                    suggested_stop_loss_pct=result.suggested_stop_loss_pct,
                    suggested_take_profit_pct=result.suggested_take_profit_pct,
                )
        return result

    def _extract_signal(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> SignalResult:
        """Extract entry signal from the last row of the processed dataframe.

        Only enter_long / enter_short produce BUY/SELL signals here.
        Exit signals (exit_long / exit_short) are for closing existing
        positions and are handled by the triple barrier (SL/TP/time-limit),
        not by emitting new entry signals.
        """
        last = dataframe.iloc[-1]

        enter_long = bool(last.get('enter_long', 0))
        enter_short = bool(last.get('enter_short', 0))

        confidence = float(last.get('signal_confidence', 0.5))
        reason = str(last.get('signal_reason', ''))

        sl_pct = (
            float(last['suggested_sl_pct'])
            if 'suggested_sl_pct' in last and pd.notna(last['suggested_sl_pct'])
            else None
        )
        tp_pct = (
            float(last['suggested_tp_pct'])
            if 'suggested_tp_pct' in last and pd.notna(last['suggested_tp_pct'])
            else None
        )

        if enter_long:
            return SignalResult(
                signal=Signal.BUY,
                confidence=confidence,
                reason=reason or 'Long entry signal',
                suggested_stop_loss_pct=sl_pct,
                suggested_take_profit_pct=tp_pct,
            )
        if enter_short:
            return SignalResult(
                signal=Signal.SELL,
                confidence=confidence,
                reason=reason or 'Short entry signal',
                suggested_stop_loss_pct=sl_pct,
                suggested_take_profit_pct=tp_pct,
            )

        return SignalResult(
            signal=Signal.HOLD,
            confidence=0.0,
            reason=reason or 'No signal',
        )


def strategy_safe_wrapper(func):
    """Wrap strategy callbacks to catch exceptions and log instead of crashing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            log.exception('strategy_error', func=func.__name__)
            # Return the dataframe unmodified if available
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    return arg
            return None

    return wrapper
