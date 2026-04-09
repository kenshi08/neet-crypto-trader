"""Tests for the SQLite database layer."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from nct.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / 'test.sqlite')
    await database.connect()
    yield database
    await database.close()


class TestDatabaseConnection:
    async def test_connect_creates_tables(self, db: Database):
        cursor = await db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        assert 'trades' in tables
        assert 'budget_periods' in tables
        assert 'daily_stats' in tables
        assert 'executor_log' in tables


class TestTradeOperations:
    async def test_insert_and_retrieve_trade(self, db: Database):
        trade_id = await db.insert_trade(
            inst_id='BTC-USDT',
            side='buy',
            size=Decimal('0.01'),
            entry_price=Decimal('67500'),
            fee=Decimal('0.675'),
            opened_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
            is_dry_run=True,
            strategy='momentum',
            signal_confidence=0.75,
            stop_loss_price=Decimal('65475'),
            take_profit_price=Decimal('70875'),
        )
        assert trade_id > 0

        open_trades = await db.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]['inst_id'] == 'BTC-USDT'
        assert open_trades[0]['closed_at'] is None

    async def test_close_trade(self, db: Database):
        trade_id = await db.insert_trade(
            inst_id='ETH-USDT',
            side='buy',
            size=Decimal('1.0'),
            entry_price=Decimal('3500'),
            fee=Decimal('3.5'),
            opened_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
            is_dry_run=True,
        )

        await db.close_trade(
            trade_id,
            exit_price=Decimal('3600'),
            pnl=Decimal('96.5'),
            closed_at=datetime(2026, 4, 9, 13, 0, tzinfo=UTC),
        )

        open_trades = await db.get_open_trades()
        assert len(open_trades) == 0


class TestBudgetPeriodOperations:
    async def test_create_and_get_active_period(self, db: Database):
        period_id = await db.create_budget_period(
            period_type='weekly',
            start_date=datetime(2026, 4, 7, tzinfo=UTC),
            end_date=datetime(2026, 4, 14, tzinfo=UTC),
        )

        period = await db.get_active_budget_period('weekly')
        assert period is not None
        assert period['id'] == period_id
        assert period['status'] == 'active'
        assert period['capital_deployed'] == '0'

    async def test_update_budget_period(self, db: Database):
        period_id = await db.create_budget_period(
            period_type='weekly',
            start_date=datetime(2026, 4, 7, tzinfo=UTC),
            end_date=datetime(2026, 4, 14, tzinfo=UTC),
        )

        await db.update_budget_period(
            period_id,
            capital_deployed=Decimal('250'),
            realized_pnl=Decimal('15.5'),
            trade_count=3,
        )

        period = await db.get_active_budget_period('weekly')
        assert period is not None
        assert period['capital_deployed'] == '250'
        assert period['realized_pnl'] == '15.5'
        assert period['trade_count'] == 3

    async def test_close_budget_period(self, db: Database):
        period_id = await db.create_budget_period(
            period_type='weekly',
            start_date=datetime(2026, 4, 7, tzinfo=UTC),
            end_date=datetime(2026, 4, 14, tzinfo=UTC),
        )

        await db.close_budget_period(period_id)

        period = await db.get_active_budget_period('weekly')
        assert period is None  # No active period after close

    async def test_no_active_period_returns_none(self, db: Database):
        period = await db.get_active_budget_period('weekly')
        assert period is None


class TestDailyStats:
    async def test_upsert_creates_and_updates(self, db: Database):
        await db.upsert_daily_stats(
            date_str='2026-04-09',
            realized_pnl=Decimal('25.50'),
            trade_count=3,
            stop_loss_count=1,
        )

        stats = await db.get_daily_stats('2026-04-09')
        assert stats is not None
        assert stats['realized_pnl'] == '25.50'
        assert stats['trade_count'] == 3

        # Update existing
        await db.upsert_daily_stats(
            date_str='2026-04-09',
            realized_pnl=Decimal('30.00'),
            trade_count=4,
            stop_loss_count=1,
        )

        stats = await db.get_daily_stats('2026-04-09')
        assert stats['realized_pnl'] == '30.00'
        assert stats['trade_count'] == 4


class TestExecutorLog:
    async def test_log_action(self, db: Database):
        await db.log_action(
            action='order_placed',
            details='BTC-USDT buy 0.01 @ 67500',
            timestamp=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
        )

        cursor = await db.conn.execute("SELECT * FROM executor_log")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]['action'] == 'order_placed'
