"""Tests for PositionSizer."""

from __future__ import annotations

from decimal import Decimal

from nct.config import BudgetConfig, RiskConfig
from nct.risk.position_sizer import PositionSizer


def _sizer(
    budget_amount: Decimal = Decimal('500'),
    max_position_pct: Decimal = Decimal('20'),
    stop_loss_pct: Decimal = Decimal('3'),
    take_profit_pct: Decimal = Decimal('5'),
) -> PositionSizer:
    return PositionSizer(
        budget_config=BudgetConfig(
            amount_usdt=budget_amount,
            max_position_pct=max_position_pct,
        ),
        risk_config=RiskConfig(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        ),
    )


class TestCalculateSize:
    def test_basic_sizing(self):
        sizer = _sizer()
        # max_position = 20% of 500 = 100 USDT, confidence=1.0
        # size = 100 / 67500 ≈ 0.00148...
        size = sizer.calculate_size(
            available_balance=Decimal('1000'),
            current_price=Decimal('67500'),
            signal_confidence=1.0,
        )
        assert size > 0
        assert size * Decimal('67500') <= Decimal('100')  # capped at 100 USDT

    def test_confidence_scales_size(self):
        sizer = _sizer()
        full = sizer.calculate_size(
            available_balance=Decimal('1000'),
            current_price=Decimal('100'),
            signal_confidence=1.0,
        )
        half = sizer.calculate_size(
            available_balance=Decimal('1000'),
            current_price=Decimal('100'),
            signal_confidence=0.5,
        )
        assert half < full
        assert half > 0

    def test_zero_balance_returns_zero(self):
        sizer = _sizer()
        size = sizer.calculate_size(
            available_balance=Decimal('0'),
            current_price=Decimal('100'),
        )
        assert size == Decimal(0)

    def test_zero_price_returns_zero(self):
        sizer = _sizer()
        size = sizer.calculate_size(
            available_balance=Decimal('1000'),
            current_price=Decimal('0'),
        )
        assert size == Decimal(0)

    def test_caps_by_available_balance(self):
        sizer = _sizer()
        # Balance is 50, less than max_position of 100
        size = sizer.calculate_size(
            available_balance=Decimal('50'),
            current_price=Decimal('100'),
            signal_confidence=1.0,
        )
        assert size * Decimal('100') <= Decimal('50')

    def test_caps_by_budget_remaining(self):
        sizer = _sizer()
        size = sizer.calculate_size(
            available_balance=Decimal('1000'),
            current_price=Decimal('100'),
            signal_confidence=1.0,
            budget_remaining=Decimal('30'),
        )
        assert size * Decimal('100') <= Decimal('30')

    def test_minimum_confidence_clamped(self):
        sizer = _sizer()
        # Confidence below 0.1 is clamped to 0.1
        size = sizer.calculate_size(
            available_balance=Decimal('1000'),
            current_price=Decimal('100'),
            signal_confidence=0.0,
        )
        assert size > 0  # clamped to 0.1, not zero


class TestStopLossPrice:
    def test_buy_stop_loss_below_entry(self):
        sizer = _sizer(stop_loss_pct=Decimal('3'))
        sl = sizer.calculate_stop_loss_price(
            entry_price=Decimal('100'), side='buy',
        )
        assert sl == Decimal('97')

    def test_sell_stop_loss_above_entry(self):
        sizer = _sizer(stop_loss_pct=Decimal('3'))
        sl = sizer.calculate_stop_loss_price(
            entry_price=Decimal('100'), side='sell',
        )
        assert sl == Decimal('103')

    def test_custom_stop_loss_pct(self):
        sizer = _sizer()
        sl = sizer.calculate_stop_loss_price(
            entry_price=Decimal('100'),
            side='buy',
            stop_loss_pct=Decimal('5'),
        )
        assert sl == Decimal('95')


class TestTakeProfitPrice:
    def test_buy_take_profit_above_entry(self):
        sizer = _sizer(take_profit_pct=Decimal('5'))
        tp = sizer.calculate_take_profit_price(
            entry_price=Decimal('100'), side='buy',
        )
        assert tp == Decimal('105')

    def test_sell_take_profit_below_entry(self):
        sizer = _sizer(take_profit_pct=Decimal('5'))
        tp = sizer.calculate_take_profit_price(
            entry_price=Decimal('100'), side='sell',
        )
        assert tp == Decimal('95')
