"""Tests for Phase 21 risk management improvements.

Covers:
- ATR-based volatility parity position sizing (#101)
- Drawdown-responsive position sizing (#102)
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from nct.config import BudgetConfig, RiskConfig
from nct.risk.position_sizer import PositionSizer

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def budget_config() -> BudgetConfig:
    return BudgetConfig(
        amount_usdt=Decimal('10000'),
        max_position_pct=Decimal('50'),
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        stop_loss_pct=Decimal('3.0'),
        take_profit_pct=Decimal('5.0'),
        use_volatility_parity=True,
        target_risk_pct=1.0,
        drawdown_scaling=True,
        drawdown_reduction_per_loss=15.0,
        drawdown_max_reduction_pct=60.0,
    )


@pytest.fixture
def sizer(budget_config, risk_config) -> PositionSizer:
    return PositionSizer(budget_config, risk_config)


# ===================================================================
# Issue #101 — ATR-based volatility parity sizing
# ===================================================================


class TestVolatilityParitySizing:
    def test_high_atr_gives_smaller_position(self, sizer):
        # High ATR asset (e.g. memecoin with 5% daily volatility)
        high_atr_size = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('5'),  # 5% of price
            signal_confidence=0.8,
        )

        # Low ATR asset (e.g. BTC with 1.5% daily volatility)
        low_atr_size = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('1.5'),  # 1.5% of price
            signal_confidence=0.8,
        )

        # Lower ATR should get larger position
        assert low_atr_size > high_atr_size

    def test_zero_atr_falls_back_to_standard(self, sizer):
        standard = sizer.calculate_size(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            signal_confidence=0.8,
        )
        atr_zero = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('0'),
            signal_confidence=0.8,
        )
        assert atr_zero == standard

    def test_none_atr_falls_back_to_standard(self, sizer):
        standard = sizer.calculate_size(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            signal_confidence=0.8,
        )
        atr_none = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=None,
            signal_confidence=0.8,
        )
        assert atr_none == standard

    def test_confidence_scales_position(self, sizer):
        full = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('2'),
            signal_confidence=1.0,
        )
        half = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('2'),
            signal_confidence=0.5,
        )
        assert half < full
        assert float(half / full) == pytest.approx(0.5, abs=0.05)

    def test_capped_by_max_position(self, sizer):
        # Very low ATR → very large position → should be capped
        size = sizer.calculate_size_with_atr(
            available_balance=Decimal('500000'),
            current_price=Decimal('100'),
            current_atr=Decimal('0.01'),  # tiny ATR
            signal_confidence=1.0,
        )
        # Max position = 10000 * 50% = $5000 → 50 units at $100
        max_size = Decimal('10000') * Decimal('50') / 100 / Decimal('100')
        assert size <= max_size

    def test_capped_by_available_balance(self, sizer):
        size = sizer.calculate_size_with_atr(
            available_balance=Decimal('50'),  # only $50
            current_price=Decimal('100'),
            current_atr=Decimal('1'),
            signal_confidence=1.0,
        )
        assert size * Decimal('100') <= Decimal('50')

    def test_capped_by_budget_remaining(self, sizer):
        size = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('1'),
            signal_confidence=1.0,
            budget_remaining=Decimal('30'),
        )
        assert size * Decimal('100') <= Decimal('30')

    def test_zero_balance_returns_zero(self, sizer):
        size = sizer.calculate_size_with_atr(
            available_balance=Decimal('0'),
            current_price=Decimal('100'),
            current_atr=Decimal('2'),
        )
        assert size == Decimal(0)


# ===================================================================
# Issue #102 — Drawdown-responsive sizing
# ===================================================================


class TestDrawdownScaling:
    def test_no_losses_returns_one(self, sizer):
        assert sizer.drawdown_scale_factor(0) == Decimal(1)

    def test_one_loss_reduces_15pct(self, sizer):
        factor = sizer.drawdown_scale_factor(1)
        assert factor == Decimal('0.85')

    def test_two_losses_reduces_30pct(self, sizer):
        factor = sizer.drawdown_scale_factor(2)
        assert factor == Decimal('0.70')

    def test_three_losses_reduces_45pct(self, sizer):
        factor = sizer.drawdown_scale_factor(3)
        assert factor == Decimal('0.55')

    def test_four_losses_hits_floor(self, sizer):
        factor = sizer.drawdown_scale_factor(4)
        assert factor == Decimal('0.40')  # 60% max reduction → 40% floor

    def test_many_losses_stays_at_floor(self, sizer):
        factor = sizer.drawdown_scale_factor(10)
        assert factor == Decimal('0.40')

    def test_drawdown_applied_to_atr_sizing(self, sizer):
        normal = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('2'),
            signal_confidence=0.8,
            consecutive_losses=0,
        )
        after_3_losses = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('2'),
            signal_confidence=0.8,
            consecutive_losses=3,
        )
        # 3 losses → 55% of normal
        ratio = float(after_3_losses / normal)
        assert 0.50 < ratio < 0.60

    def test_drawdown_disabled_no_effect(self):
        risk = RiskConfig(
            drawdown_scaling=False,
            use_volatility_parity=True,
            target_risk_pct=1.0,
        )
        budget = BudgetConfig(amount_usdt=Decimal('1000'))
        sizer = PositionSizer(budget, risk)

        normal = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('2'),
            consecutive_losses=0,
        )
        with_losses = sizer.calculate_size_with_atr(
            available_balance=Decimal('5000'),
            current_price=Decimal('100'),
            current_atr=Decimal('2'),
            consecutive_losses=5,
        )
        assert normal == with_losses
