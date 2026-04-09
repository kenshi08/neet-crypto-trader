"""Tests for BudgetManager — the most safety-critical component."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from nct.config import BudgetConfig
from nct.db import Database
from nct.risk.budget_manager import BudgetManager


@pytest.fixture
def budget_config() -> BudgetConfig:
    return BudgetConfig(
        period='weekly',
        amount_usdt=Decimal('500'),
        max_loss_pct=Decimal('5.0'),     # -25 USDT
        max_gain_pct=Decimal('15.0'),    # +75 USDT
        max_position_pct=Decimal('20.0'),  # 100 USDT per trade
        daily_loss_limit_usdt=Decimal('50'),
    )


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / 'test_budget.sqlite')
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def manager(budget_config: BudgetConfig, db: Database) -> BudgetManager:
    mgr = BudgetManager(budget_config, db)
    await mgr.initialize()
    return mgr


class TestCanOpenTrade:
    async def test_allows_trade_within_budget(self, manager: BudgetManager):
        allowed, reason = manager.can_open_trade(Decimal('80'))
        assert allowed is True
        assert reason == ''

    async def test_denies_when_budget_exhausted(self, manager: BudgetManager):
        await manager.record_trade_open(Decimal('450'))
        allowed, reason = manager.can_open_trade(Decimal('60'))
        assert allowed is False
        assert 'Budget exhausted' in reason

    async def test_denies_when_position_too_large(self, manager: BudgetManager):
        # max_position_pct=20% of 500 = 100 USDT max per trade
        allowed, reason = manager.can_open_trade(Decimal('150'))
        assert allowed is False
        assert 'Position too large' in reason

    async def test_allows_position_at_exact_limit(self, manager: BudgetManager):
        # 20% of 500 = 100 USDT exactly
        allowed, _reason = manager.can_open_trade(Decimal('100'))
        assert allowed is True

    async def test_denies_when_daily_loss_limit_hit(self, manager: BudgetManager):
        await manager.record_trade_close(Decimal('-50'))
        allowed, reason = manager.can_open_trade(Decimal('50'))
        assert allowed is False
        assert 'Daily loss limit' in reason

    async def test_denies_when_weekly_loss_limit_hit(self, manager: BudgetManager):
        # max_loss_pct=5% of 500 = 25 USDT
        await manager.record_trade_close(Decimal('-25'))
        allowed, reason = manager.can_open_trade(Decimal('50'))
        assert allowed is False
        assert 'loss limit hit' in reason

    async def test_denies_when_weekly_gain_target_hit(self, manager: BudgetManager):
        # max_gain_pct=15% of 500 = 75 USDT
        await manager.record_trade_close(Decimal('75'))
        allowed, reason = manager.can_open_trade(Decimal('50'))
        assert allowed is False
        assert 'gain target reached' in reason

    async def test_budget_exactly_at_loss_limit(self, manager: BudgetManager):
        # Exactly at -25 (5% of 500) — should deny
        await manager.record_trade_close(Decimal('-25'))
        allowed, _ = manager.can_open_trade(Decimal('10'))
        assert allowed is False

    async def test_budget_one_below_loss_limit(self, manager: BudgetManager):
        # At -24.99 — still allowed
        await manager.record_trade_close(Decimal('-24.99'))
        allowed, _ = manager.can_open_trade(Decimal('10'))
        assert allowed is True


class TestRecordTrade:
    async def test_record_open_updates_capital(self, manager: BudgetManager):
        await manager.record_trade_open(Decimal('100'))
        assert manager.capital_deployed == Decimal('100')
        assert manager.budget_remaining == Decimal('400')

    async def test_record_close_updates_pnl(self, manager: BudgetManager):
        await manager.record_trade_close(Decimal('15'))
        assert manager.realized_pnl == Decimal('15')
        assert manager.daily_pnl == Decimal('15')

    async def test_record_stop_loss(self, manager: BudgetManager):
        await manager.record_trade_close(Decimal('-10'), was_stop_loss=True)
        assert manager.daily_stop_loss_count == 1

    async def test_multiple_trades_accumulate(self, manager: BudgetManager):
        await manager.record_trade_open(Decimal('100'))
        await manager.record_trade_open(Decimal('100'))
        assert manager.capital_deployed == Decimal('200')
        assert manager.trade_count == 2

        await manager.record_trade_close(Decimal('10'))
        await manager.record_trade_close(Decimal('-5'))
        assert manager.realized_pnl == Decimal('5')
        assert manager.daily_pnl == Decimal('5')


class TestLimitChecks:
    async def test_is_daily_loss_limit_hit(self, manager: BudgetManager):
        assert manager.is_daily_loss_limit_hit() is False
        await manager.record_trade_close(Decimal('-50'))
        assert manager.is_daily_loss_limit_hit() is True

    async def test_is_period_loss_limit_hit(self, manager: BudgetManager):
        assert manager.is_period_loss_limit_hit() is False
        await manager.record_trade_close(Decimal('-25'))
        assert manager.is_period_loss_limit_hit() is True

    async def test_is_period_gain_target_hit(self, manager: BudgetManager):
        assert manager.is_period_gain_target_hit() is False
        await manager.record_trade_close(Decimal('75'))
        assert manager.is_period_gain_target_hit() is True


class TestPersistence:
    async def test_survives_restart(self, budget_config: BudgetConfig, db: Database):
        # First session
        mgr1 = BudgetManager(budget_config, db)
        await mgr1.initialize()
        await mgr1.record_trade_open(Decimal('200'))
        await mgr1.record_trade_close(Decimal('15'))

        # Second session — same DB
        mgr2 = BudgetManager(budget_config, db)
        await mgr2.initialize()
        assert mgr2.capital_deployed == Decimal('200')
        assert mgr2.realized_pnl == Decimal('15')
        assert mgr2.trade_count == 1


class TestPeriodReset:
    async def test_expired_period_resets_on_init(self, budget_config: BudgetConfig, db: Database):
        # Create an expired period
        await db.create_budget_period(
            period_type='weekly',
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2020, 1, 8, tzinfo=UTC),
        )

        mgr = BudgetManager(budget_config, db)
        await mgr.initialize()

        # Should have started a fresh period
        assert mgr.capital_deployed == Decimal(0)
        assert mgr.realized_pnl == Decimal(0)
