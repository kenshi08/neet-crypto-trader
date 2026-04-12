"""Tests for position reconciliation auto-fix."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from nct.exchange.models import OrderStatus, Position, Ticker
from nct.portfolio.tracker import PortfolioTracker, TrackedTrade


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.is_demo = True
    client.get_positions = AsyncMock(return_value=[])
    client.get_ticker = AsyncMock(return_value=Ticker(
        inst_id='BTC-USD', last=Decimal('65000'), bid=Decimal('64990'),
        ask=Decimal('65010'), bid_size=Decimal('1'), ask_size=Decimal('1'),
        volume_24h=Decimal('1000'),
        timestamp=MagicMock(),
    ))
    return client


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.get_open_trades = AsyncMock(return_value=[])
    db.insert_trade = AsyncMock(return_value=1)
    db.close_trade = AsyncMock()
    db.log_action = AsyncMock()
    return db


@pytest.fixture
def tracker(mock_client, mock_db):
    return PortfolioTracker(mock_client, mock_db)


class TestReconcileStaleTradesRemoval:
    @pytest.mark.asyncio
    async def test_stale_trade_auto_closed(self, tracker, mock_client):
        """If exchange shows positions but not BTC-USD, close the stale trade."""
        tracker._open_trades['BTC-USD'] = TrackedTrade(
            trade_id=1, inst_id='BTC-USD', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('64000'),
            fee=Decimal('0.1'),
        )
        # Also track ETH-USD so it doesn't appear as untracked
        tracker._open_trades['ETH-USD'] = TrackedTrade(
            trade_id=2, inst_id='ETH-USD', side='buy',
            size=Decimal('0.1'), entry_price=Decimal('3000'),
            fee=Decimal('0.01'),
        )

        # Exchange shows ETH-USD but NOT BTC-USD
        mock_client.get_positions.return_value = [
            Position(inst_id='ETH-USD', side='long', size=Decimal('0.1'),
                     avg_price=Decimal('3000')),
        ]

        events = await tracker.reconcile()

        stale_events = [(i, e) for i, e in events if e == 'stale_closed']
        assert len(stale_events) == 1
        assert stale_events[0][0] == 'BTC-USD'
        assert 'BTC-USD' not in tracker._open_trades

    @pytest.mark.asyncio
    async def test_no_stale_if_exchange_returns_empty(self, tracker, mock_client):
        """Don't auto-close if exchange returns no positions (e.g., Coinbase spot)."""
        tracker._open_trades['BTC-USD'] = TrackedTrade(
            trade_id=1, inst_id='BTC-USD', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('64000'),
            fee=Decimal('0.1'),
        )
        mock_client.get_positions.return_value = []

        events = await tracker.reconcile()

        # Should NOT auto-close — empty response means no position data available
        assert len(events) == 0
        assert 'BTC-USD' in tracker._open_trades


class TestReconcileUntrackedDetection:
    @pytest.mark.asyncio
    async def test_untracked_position_detected(self, tracker, mock_client):
        """Positions on exchange but not in DB should be flagged."""
        mock_client.get_positions.return_value = [
            Position(inst_id='SOL-USD', side='long', size=Decimal('10'),
                     avg_price=Decimal('150')),
        ]

        events = await tracker.reconcile()

        assert len(events) == 1
        assert events[0] == ('SOL-USD', 'untracked_detected')
        # Should NOT be adopted into tracker
        assert 'SOL-USD' not in tracker._open_trades


class TestReconcileNoIssues:
    @pytest.mark.asyncio
    async def test_matching_positions_no_events(self, tracker, mock_client):
        """When DB and exchange agree, no events should fire."""
        tracker._open_trades['BTC-USD'] = TrackedTrade(
            trade_id=1, inst_id='BTC-USD', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('64000'),
            fee=Decimal('0.1'),
        )
        mock_client.get_positions.return_value = [
            Position(inst_id='BTC-USD', side='long', size=Decimal('0.01'),
                     avg_price=Decimal('64000')),
        ]

        events = await tracker.reconcile()
        assert len(events) == 0
        assert 'BTC-USD' in tracker._open_trades

    @pytest.mark.asyncio
    async def test_reconcile_handles_exception(self, tracker, mock_client):
        """If get_positions fails, return empty events (no crash)."""
        mock_client.get_positions.side_effect = Exception('API error')

        events = await tracker.reconcile()
        assert events == []
