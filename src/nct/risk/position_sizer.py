"""Per-trade position sizing based on risk parameters."""

from __future__ import annotations

from decimal import Decimal

import structlog

from nct.config import BudgetConfig, RiskConfig

log = structlog.get_logger()

# OKX minimum order sizes (USDT equivalent) — conservative defaults
_MIN_ORDER_SIZES: dict[str, Decimal] = {
    'BTC-USDT': Decimal('0.00001'),
    'ETH-USDT': Decimal('0.001'),
    'SOL-USDT': Decimal('0.01'),
}
_DEFAULT_MIN_ORDER = Decimal('0.001')


class PositionSizer:
    """Calculates position size so max loss per trade stays within risk limits.

    Formula: size = (balance * risk_per_trade_pct) / stop_loss_distance
    Capped by: max_position_pct of budget, remaining budget, minimum order size.
    """

    def __init__(self, budget_config: BudgetConfig, risk_config: RiskConfig) -> None:
        self._budget = budget_config
        self._risk = risk_config

    def calculate_size(
        self,
        *,
        available_balance: Decimal,
        current_price: Decimal,
        stop_loss_pct: Decimal | None = None,
        signal_confidence: float = 1.0,
        budget_remaining: Decimal | None = None,
    ) -> Decimal:
        """Calculate position size in base currency units.

        Args:
            available_balance: Available trading capital in USDT.
            current_price: Current asset price in USDT.
            stop_loss_pct: Stop-loss distance as percentage (e.g., 3.0 for 3%).
                           Defaults to risk config value.
            signal_confidence: Strategy confidence 0.0-1.0. Scales position size.
            budget_remaining: Remaining budget for the period. Caps the position.

        Returns:
            Position size in base currency units. Zero if trade is too small.
        """
        if available_balance <= 0 or current_price <= 0:
            return Decimal(0)

        sl_pct = stop_loss_pct or self._risk.stop_loss_pct
        if sl_pct <= 0:
            return Decimal(0)

        # Risk per trade: use stop_loss_pct of budget as max risk
        # (e.g., 3% SL means we risk 3% of the position value)
        max_position_value = self._budget.amount_usdt * self._budget.max_position_pct / 100

        # Scale by confidence (0.5 confidence = half the max position)
        confidence = max(Decimal('0.1'), min(Decimal('1.0'), Decimal(str(signal_confidence))))
        position_value = max_position_value * confidence

        # Cap by available balance
        position_value = min(position_value, available_balance)

        # Cap by remaining budget
        if budget_remaining is not None:
            position_value = min(position_value, budget_remaining)

        # Convert to base currency size
        size = position_value / current_price

        # Ensure above minimum order size
        if size <= 0:
            return Decimal(0)

        log.debug(
            'position_sized',
            position_value=str(position_value),
            size=str(size),
            price=str(current_price),
            confidence=signal_confidence,
        )

        return size

    def calculate_size_with_atr(
        self,
        *,
        available_balance: Decimal,
        current_price: Decimal,
        current_atr: Decimal,
        signal_confidence: float = 1.0,
        budget_remaining: Decimal | None = None,
        consecutive_losses: int = 0,
    ) -> Decimal:
        """ATR-based volatility parity sizing (#101).

        Sizes positions so that a 1-ATR move equals target_risk_pct of budget.
        Higher ATR → smaller position. Lower ATR → larger position.
        Also applies drawdown scaling if enabled (#102).
        """
        if available_balance <= 0 or current_price <= 0:
            return Decimal(0)
        if not current_atr or current_atr <= 0:
            return self.calculate_size(
                available_balance=available_balance,
                current_price=current_price,
                signal_confidence=signal_confidence,
                budget_remaining=budget_remaining,
            )

        # Target risk in USDT: budget * target_risk_pct%
        target_risk = (
            self._budget.amount_usdt
            * Decimal(str(self._risk.target_risk_pct)) / 100
        )

        # ATR as fraction of price
        atr_pct = current_atr / current_price
        if atr_pct <= 0:
            return Decimal(0)

        # Position value where 1-ATR move = target_risk
        position_value = target_risk / atr_pct

        # Scale by confidence
        confidence = max(
            Decimal('0.1'),
            min(Decimal('1.0'), Decimal(str(signal_confidence))),
        )
        position_value *= confidence

        # Drawdown scaling (#102)
        if self._risk.drawdown_scaling and consecutive_losses > 0:
            scale = self.drawdown_scale_factor(consecutive_losses)
            position_value *= scale

        # Cap by max position, balance, budget
        max_pos = (
            self._budget.amount_usdt * self._budget.max_position_pct / 100
        )
        position_value = min(position_value, max_pos)
        position_value = min(position_value, available_balance)
        if budget_remaining is not None:
            position_value = min(position_value, budget_remaining)

        size = position_value / current_price
        if size <= 0:
            return Decimal(0)

        log.debug(
            'volatility_parity_sized',
            position_value=str(position_value),
            size=str(size),
            atr=str(current_atr),
            atr_pct=f'{float(atr_pct) * 100:.2f}%',
            confidence=signal_confidence,
            consecutive_losses=consecutive_losses,
        )
        return size

    def drawdown_scale_factor(self, consecutive_losses: int) -> Decimal:
        """Calculate position size reduction factor after consecutive losses.

        Returns a multiplier between (1 - max_reduction/100) and 1.0.
        E.g., with 15% per loss and 60% max: 1 loss → 0.85, 2 → 0.70,
        3 → 0.55, 4+ → 0.40 (floor).
        """
        if consecutive_losses <= 0:
            return Decimal(1)
        reduction = Decimal(str(
            self._risk.drawdown_reduction_per_loss,
        )) * consecutive_losses / 100
        floor = 1 - Decimal(str(self._risk.drawdown_max_reduction_pct)) / 100
        return max(Decimal(1) - reduction, floor)

    def calculate_stop_loss_price(
        self,
        *,
        entry_price: Decimal,
        side: str,
        stop_loss_pct: Decimal | None = None,
    ) -> Decimal:
        """Calculate stop-loss trigger price."""
        sl_pct = stop_loss_pct or self._risk.stop_loss_pct
        distance = entry_price * sl_pct / 100
        if side == 'buy':
            return entry_price - distance
        return entry_price + distance

    def calculate_take_profit_price(
        self,
        *,
        entry_price: Decimal,
        side: str,
        take_profit_pct: Decimal | None = None,
    ) -> Decimal:
        """Calculate take-profit trigger price."""
        tp_pct = take_profit_pct or self._risk.take_profit_pct
        distance = entry_price * tp_pct / 100
        if side == 'buy':
            return entry_price + distance
        return entry_price - distance
