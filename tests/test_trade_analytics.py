"""Tests for trade analytics schema, recording, and aggregation."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from nct.db import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / 'test.sqlite')
    await database.connect()
    yield database
    await database.close()


class TestTradeAnalyticsRecordOpen:
    @pytest.mark.asyncio
    async def test_record_open(self, db):
        # First insert a trade (FK requirement)
        trade_id = await db.insert_trade(
            inst_id='BTC-USD', side='buy', size=Decimal('0.01'),
            entry_price=Decimal('65000'), fee=Decimal('2.6'),
            opened_at=datetime.now(UTC), is_dry_run=True,
        )

        await db.record_trade_analytics_open(
            trade_id=trade_id,
            inst_id='BTC-USD',
            strategy='momentum',
            signal_confidence=0.75,
            intended_entry_price='65000',
            actual_entry_price='65010',
            entry_slippage_pct=0.015,
            intended_size='0.01',
            actual_filled_size='0.01',
            fill_rate=1.0,
            entry_latency_ms=145.3,
            sl_placement_latency_ms=89.1,
            tp_placement_latency_ms=92.7,
            fees_paid='2.6',
            opened_at=datetime.now(UTC),
        )

        cursor = await db.conn.execute(
            "SELECT * FROM trade_analytics WHERE trade_id = ?", (trade_id,)
        )
        row = dict(await cursor.fetchone())
        assert row['inst_id'] == 'BTC-USD'
        assert row['strategy'] == 'momentum'
        assert row['signal_confidence'] == 0.75
        assert row['entry_slippage_pct'] == pytest.approx(0.015)
        assert row['entry_latency_ms'] == pytest.approx(145.3)
        assert row['fill_rate'] == pytest.approx(1.0)


class TestTradeAnalyticsRecordClose:
    @pytest.mark.asyncio
    async def test_record_close(self, db):
        trade_id = await db.insert_trade(
            inst_id='ETH-USD', side='buy', size=Decimal('0.1'),
            entry_price=Decimal('3000'), fee=Decimal('1.2'),
            opened_at=datetime.now(UTC), is_dry_run=True,
        )
        await db.record_trade_analytics_open(
            trade_id=trade_id, inst_id='ETH-USD',
            opened_at=datetime.now(UTC),
        )

        now = datetime.now(UTC)
        await db.record_trade_analytics_close(
            trade_id=trade_id,
            exit_reason='sl_hit',
            exit_slippage_pct=0.02,
            closed_at=now,
        )

        cursor = await db.conn.execute(
            "SELECT exit_reason, exit_slippage_pct FROM trade_analytics WHERE trade_id = ?",
            (trade_id,),
        )
        row = dict(await cursor.fetchone())
        assert row['exit_reason'] == 'sl_hit'
        assert row['exit_slippage_pct'] == pytest.approx(0.02)


class TestTradeAnalyticsSummary:
    @pytest.mark.asyncio
    async def test_summary_aggregation(self, db):
        # Insert 2 trades with analytics
        for inst, pnl_val in [('BTC-USD', '10'), ('ETH-USD', '-5')]:
            tid = await db.insert_trade(
                inst_id=inst, side='buy', size=Decimal('0.01'),
                entry_price=Decimal('1000'), fee=Decimal('0.5'),
                opened_at=datetime.now(UTC), is_dry_run=True,
            )
            await db.close_trade(
                tid, exit_price=Decimal('1000'), pnl=Decimal(pnl_val),
                closed_at=datetime.now(UTC),
            )
            await db.record_trade_analytics_open(
                trade_id=tid, inst_id=inst, entry_slippage_pct=0.01,
                entry_latency_ms=100, fees_paid='0.5',
                opened_at=datetime.now(UTC),
            )
            await db.record_trade_analytics_close(
                trade_id=tid, exit_reason='tp_hit',
                closed_at=datetime.now(UTC),
            )

        summary = await db.get_trade_analytics_summary()
        assert summary['total_trades'] == 2
        assert summary['wins'] == 1
        assert summary['losses'] == 1
        assert summary['total_pnl'] == pytest.approx(5.0)
        assert summary['avg_slippage'] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_summary_empty(self, db):
        summary = await db.get_trade_analytics_summary()
        assert summary.get('total_trades', 0) == 0


class TestTradeAnalyticsByPair:
    @pytest.mark.asyncio
    async def test_per_pair_breakdown(self, db):
        for inst, pnl_val in [('BTC-USD', '20'), ('BTC-USD', '-3'), ('ETH-USD', '5')]:
            tid = await db.insert_trade(
                inst_id=inst, side='buy', size=Decimal('0.01'),
                entry_price=Decimal('1000'), fee=Decimal('0.1'),
                opened_at=datetime.now(UTC), is_dry_run=True,
            )
            await db.close_trade(
                tid, exit_price=Decimal('1000'), pnl=Decimal(pnl_val),
                closed_at=datetime.now(UTC),
            )
            await db.record_trade_analytics_open(
                trade_id=tid, inst_id=inst, opened_at=datetime.now(UTC),
            )

        by_pair = await db.get_trade_analytics_by_pair()
        assert len(by_pair) == 2
        # BTC-USD has higher total P&L, should be first
        assert by_pair[0]['inst_id'] == 'BTC-USD'
        assert by_pair[0]['trades'] == 2
        assert by_pair[0]['pnl'] == pytest.approx(17.0)


class TestTradeAnalyticsByExitReason:
    @pytest.mark.asyncio
    async def test_per_exit_reason_breakdown(self, db):
        for reason, pnl_val in [('sl_hit', '-5'), ('tp_hit', '10'), ('tp_hit', '8')]:
            tid = await db.insert_trade(
                inst_id='BTC-USD', side='buy', size=Decimal('0.01'),
                entry_price=Decimal('1000'), fee=Decimal('0.1'),
                opened_at=datetime.now(UTC), is_dry_run=True,
            )
            await db.close_trade(
                tid, exit_price=Decimal('1000'), pnl=Decimal(pnl_val),
                closed_at=datetime.now(UTC),
            )
            await db.record_trade_analytics_open(
                trade_id=tid, inst_id='BTC-USD', opened_at=datetime.now(UTC),
            )
            await db.record_trade_analytics_close(
                trade_id=tid, exit_reason=reason, closed_at=datetime.now(UTC),
            )

        by_exit = await db.get_trade_analytics_by_exit_reason()
        assert len(by_exit) == 2
        tp_row = next(r for r in by_exit if r['exit_reason'] == 'tp_hit')
        assert tp_row['trades'] == 2
        assert tp_row['pnl'] == pytest.approx(18.0)
