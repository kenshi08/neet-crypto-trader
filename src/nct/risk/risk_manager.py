"""Central risk gate — every trade must pass through here."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import structlog

from nct.config import RiskConfig, TradingConfig
from nct.risk.budget_manager import BudgetManager
from nct.risk.position_sizer import PositionSizer
from nct.risk.protections import ProtectionManager

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class TradeDecision:
    approved: bool
    size: Decimal = Decimal(0)
    stop_loss_price: Decimal = Decimal(0)
    take_profit_price: Decimal = Decimal(0)
    time_limit_seconds: int = 0
    denial_reason: str = ''


class RiskManager:
    """Orchestrates all risk checks before a trade can execute.

    Check order:
    1. Protection plugins (circuit breakers)
    2. Budget limits (daily/weekly/monthly)
    3. Max open positions
    4. Signal confidence threshold
    5. Position sizing
    6. SL/TP/time-limit price calculation
    """

    def __init__(
        self,
        *,
        budget_manager: BudgetManager,
        position_sizer: PositionSizer,
        protection_manager: ProtectionManager,
        risk_config: RiskConfig,
        trading_config: TradingConfig,
        portfolio=None,
    ) -> None:
        self._budget = budget_manager
        self._sizer = position_sizer
        self._protections = protection_manager
        self._risk = risk_config
        self._trading = trading_config
        self._portfolio = portfolio  # PortfolioTracker for correlation checks

    def evaluate_trade(
        self,
        *,
        inst_id: str,
        side: str,
        current_price: Decimal,
        available_balance: Decimal,
        signal_confidence: float,
        open_position_count: int,
    ) -> TradeDecision:
        """Evaluate whether a trade should proceed.

        Returns a TradeDecision with approved=True and calculated sizes/prices,
        or approved=False with a denial_reason.
        """

        # 1. Protection plugins (circuit breakers)
        lock = self._protections.check(pair=inst_id)
        if lock.locked:
            return self._deny(lock.reason)

        # 1b. Correlation-aware position limits
        corr_deny = self._check_correlation(inst_id)
        if corr_deny:
            return self._deny(corr_deny)

        # 2. Signal confidence threshold
        if signal_confidence < self._risk.min_signal_confidence:
            return self._deny(
                f'Signal confidence {signal_confidence:.2f} '
                f'< minimum {self._risk.min_signal_confidence}'
            )

        # 3. Max open positions
        if open_position_count >= self._trading.max_open_positions:
            return self._deny(
                f'Max open positions reached: {open_position_count} '
                f'>= {self._trading.max_open_positions}'
            )

        # 4. Calculate position size
        size = self._sizer.calculate_size(
            available_balance=available_balance,
            current_price=current_price,
            signal_confidence=signal_confidence,
            budget_remaining=self._budget.budget_remaining,
        )
        if size <= 0:
            return self._deny('Position size calculated as zero')

        cost_usdt = size * current_price

        # 5. Budget check (must be after size calculation)
        allowed, reason = self._budget.can_open_trade(cost_usdt)
        if not allowed:
            return self._deny(reason)

        # 6. Calculate SL/TP prices
        stop_loss_price = self._sizer.calculate_stop_loss_price(
            entry_price=current_price, side=side,
        )
        take_profit_price = self._sizer.calculate_take_profit_price(
            entry_price=current_price, side=side,
        )

        decision = TradeDecision(
            approved=True,
            size=size,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            time_limit_seconds=self._risk.time_limit_seconds,
        )

        log.info(
            'trade_approved',
            inst_id=inst_id,
            side=side,
            size=str(size),
            cost_usdt=str(cost_usdt),
            stop_loss=str(stop_loss_price),
            take_profit=str(take_profit_price),
            confidence=signal_confidence,
        )

        return decision

    def _check_correlation(self, inst_id: str) -> str:
        """Check if opening this pair would exceed correlation group limits.

        Returns denial reason string if blocked, empty string if OK.
        """
        groups = self._risk.correlation_groups
        max_corr = self._risk.max_correlated_positions
        if not groups or not self._portfolio:
            return ''

        open_pairs = set(self._portfolio.open_trades.keys())

        for group_name, members in groups.items():
            if inst_id not in members:
                continue
            # Count how many from this group are already open
            open_in_group = open_pairs & set(members)
            if len(open_in_group) >= max_corr:
                return (
                    f'Correlation limit: {len(open_in_group)}/{max_corr} '
                    f'{group_name} positions open ({", ".join(sorted(open_in_group))})'
                )

        return ''

    @staticmethod
    def _deny(reason: str) -> TradeDecision:
        log.info('trade_denied', reason=reason)
        return TradeDecision(approved=False, denial_reason=reason)
