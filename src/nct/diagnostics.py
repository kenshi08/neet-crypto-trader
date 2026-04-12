"""Diagnostic engine — explains why a trade would or would not be approved.

Used by the /why and /signal Telegram commands to show per-gate pass/fail
results and current indicator values without actually placing any trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd
import structlog

from nct.config import RiskConfig, TradingConfig
from nct.portfolio.tracker import PortfolioTracker
from nct.risk.budget_manager import BudgetManager
from nct.risk.position_sizer import PositionSizer
from nct.risk.protections import ProtectionManager
from nct.strategy.base import IStrategy, Signal, SignalResult
from nct.strategy.data_provider import DataProvider

log = structlog.get_logger()

# Indicator columns we try to capture from the strategy DataFrame
_INDICATOR_COLS = (
    'close', 'rsi', 'macd', 'macd_signal', 'macd_hist',
    'atr', 'ema_fast', 'ema_slow',
    'bb_upper', 'bb_lower', 'bb_middle',
    'volume',
)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of a single risk gate check."""

    name: str     # e.g. 'protections', 'confidence', 'max_positions'
    passed: bool
    detail: str   # human-readable: 'confidence 0.72 >= 0.60'


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Full diagnostic report for a single pair."""

    inst_id: str
    signal: Signal
    confidence: float
    reason: str
    gates: list[GateResult]
    would_approve: bool
    indicators: dict[str, float]


class DiagnosticEngine:
    """Runs the full signal + risk pipeline and collects per-gate results.

    This mirrors the check order in RiskManager.evaluate_trade() but captures
    intermediate results instead of short-circuiting on first failure.
    """

    def __init__(
        self,
        *,
        strategy: IStrategy,
        risk_config: RiskConfig,
        trading_config: TradingConfig,
        budget_manager: BudgetManager,
        position_sizer: PositionSizer,
        protection_manager: ProtectionManager,
        data_provider: DataProvider,
        portfolio: PortfolioTracker,
    ) -> None:
        self._strategy = strategy
        self._risk = risk_config
        self._trading = trading_config
        self._budget = budget_manager
        self._sizer = position_sizer
        self._protections = protection_manager
        self._data = data_provider
        self._portfolio = portfolio

    async def diagnose(
        self,
        inst_id: str,
        timeframe: str,
        *,
        current_price: Decimal,
        available_balance: Decimal,
    ) -> DiagnosticReport:
        """Run full signal + risk pipeline, collecting per-gate pass/fail."""

        # 1. Fetch candles and run strategy
        signal_result, indicators = await self._run_strategy(inst_id, timeframe)

        # 2. Run each risk gate individually (don't short-circuit)
        gates: list[GateResult] = []
        all_passed = True

        # Gate 1: Protections (circuit breakers)
        lock = self._protections.check(pair=inst_id)
        passed = not lock.locked
        gates.append(GateResult(
            name='protections',
            passed=passed,
            detail='no locks active' if passed else lock.reason,
        ))
        if not passed:
            all_passed = False

        # Gate 2: Signal confidence
        conf = signal_result.confidence
        min_conf = self._risk.min_signal_confidence
        passed = conf >= min_conf
        cmp = '\u2265' if passed else '<'
        gates.append(GateResult(
            name='confidence',
            passed=passed,
            detail=f'{conf:.2f} {cmp} {min_conf}',
        ))
        if not passed:
            all_passed = False

        # Gate 3: Max open positions
        open_count = self._portfolio.open_trade_count
        max_pos = self._trading.max_open_positions
        passed = open_count < max_pos
        gates.append(GateResult(
            name='max_positions',
            passed=passed,
            detail=f'{open_count}/{max_pos}',
        ))
        if not passed:
            all_passed = False

        # Gate 4: Position sizing
        size = self._sizer.calculate_size(
            available_balance=available_balance,
            current_price=current_price,
            signal_confidence=signal_result.confidence,
            budget_remaining=self._budget.budget_remaining,
        )
        passed = size > 0
        gates.append(GateResult(
            name='sizing',
            passed=passed,
            detail=f'{size} units' if passed else 'calculated as zero',
        ))
        if not passed:
            all_passed = False

        # Gate 5: Budget check
        cost = size * current_price if size > 0 else Decimal(0)
        budget_ok, budget_reason = self._budget.can_open_trade(cost)
        gates.append(GateResult(
            name='budget',
            passed=budget_ok,
            detail='within limits' if budget_ok else budget_reason,
        ))
        if not budget_ok:
            all_passed = False

        return DiagnosticReport(
            inst_id=inst_id,
            signal=signal_result.signal,
            confidence=signal_result.confidence,
            reason=signal_result.reason,
            gates=gates,
            would_approve=all_passed,
            indicators=indicators,
        )

    async def _run_strategy(
        self, inst_id: str, timeframe: str,
    ) -> tuple[SignalResult, dict[str, float]]:
        """Run strategy pipeline and extract signal + indicator snapshot."""
        df = await self._data.get_dataframe(inst_id, timeframe)

        if len(df) < self._strategy.required_candle_count:
            return (
                SignalResult(
                    signal=Signal.HOLD,
                    confidence=0.0,
                    reason=f'Insufficient candles: {len(df)} < {self._strategy.required_candle_count}',
                ),
                _extract_indicators(df),
            )

        df = df.copy()
        metadata: dict = {'pair': inst_id, 'timeframe': timeframe}
        df = self._strategy.populate_indicators(df, metadata)
        df = self._strategy.populate_entry_trend(df, metadata)

        signal = self._strategy._extract_signal(df, metadata)
        indicators = _extract_indicators(df)

        return signal, indicators


def _extract_indicators(df: pd.DataFrame) -> dict[str, float]:
    """Extract known indicator values from the last row of a DataFrame."""
    if len(df) == 0:
        return {}

    last = df.iloc[-1]
    result: dict[str, float] = {}
    for col in _INDICATOR_COLS:
        if col in last.index and pd.notna(last[col]):
            result[col] = float(last[col])
    return result
