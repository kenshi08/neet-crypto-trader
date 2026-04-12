"""Tests for PortfolioTracker."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nct.config import OKXCredentials
from nct.db import Database
from nct.exchange.client import OKXClient
from nct.portfolio.tracker import PortfolioTracker, TrackedTrade


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / 'test_portfolio.sqlite')
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def mock_client() -> OKXClient:
    client = OKXClient(OKXCredentials(demo_mode=True))
    client._market_api = MagicMock()
    client._trade_api = MagicMock()
    client._account_api = MagicMock()
    return client


@pytest.fixture
async def tracker(mock_client: OKXClient, db: Database) -> PortfolioTracker:
    t = PortfolioTracker(mock_client, db)
    await t.initialize()
    return t


class TestTrackedTrade:
    def test_cost_usdt(self):
        trade = TrackedTrade(
            exchange='', trade_id=1, inst_id='BTC-USDT', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('67500'),
            fee=Decimal('0.675'),
        )
        assert trade.cost_usdt == Decimal('675')

    def test_unrealized_pnl_buy_profit(self):
        trade = TrackedTrade(
            exchange='', trade_id=1, inst_id='BTC-USDT', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('67500'),
            fee=Decimal('0.675'),
        )
        pnl = trade.unrealized_pnl(Decimal('68000'))
        # (68000 - 67500) * 0.01 - 0.675 = 5 - 0.675 = 4.325
        assert pnl == Decimal('4.325')

    def test_unrealized_pnl_buy_loss(self):
        trade = TrackedTrade(
            exchange='', trade_id=1, inst_id='BTC-USDT', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('67500'),
            fee=Decimal('0.675'),
        )
        pnl = trade.unrealized_pnl(Decimal('67000'))
        # (67000 - 67500) * 0.01 - 0.675 = -5 - 0.675 = -5.675
        assert pnl == Decimal('-5.675')

    def test_unrealized_pnl_sell_profit(self):
        trade = TrackedTrade(
            exchange='', trade_id=1, inst_id='BTC-USDT', side='sell',
            size=Decimal('0.01'), entry_price=Decimal('67500'),
            fee=Decimal('0.675'),
        )
        pnl = trade.unrealized_pnl(Decimal('67000'))
        # (67500 - 67000) * 0.01 - 0.675 = 5 - 0.675 = 4.325
        assert pnl == Decimal('4.325')


class TestPortfolioTracker:
    async def test_open_trade(self, tracker: PortfolioTracker):
        trade = await tracker.open_trade(
            inst_id='BTC-USDT',
            side='buy',
            size=Decimal('0.01'),
            entry_price=Decimal('67500'),
            fee=Decimal('0.675'),
        )

        assert trade.inst_id == 'BTC-USDT'
        assert trade.trade_id > 0
        assert tracker.open_trade_count == 1
        assert tracker.has_open_trade('BTC-USDT')

    async def test_close_trade(self, tracker: PortfolioTracker):
        await tracker.open_trade(
            inst_id='BTC-USDT',
            side='buy',
            size=Decimal('0.01'),
            entry_price=Decimal('67500'),
            fee=Decimal('0.675'),
        )

        pnl = await tracker.close_trade('BTC-USDT', exit_price=Decimal('68000'))

        assert pnl == Decimal('4.325')
        assert tracker.open_trade_count == 0
        assert not tracker.has_open_trade('BTC-USDT')

    async def test_close_nonexistent_trade(self, tracker: PortfolioTracker):
        pnl = await tracker.close_trade('DOGE-USDT', exit_price=Decimal('1'))
        assert pnl == Decimal(0)

    async def test_multiple_open_trades(self, tracker: PortfolioTracker):
        await tracker.open_trade(
            inst_id='BTC-USDT', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('67500'),
            fee=Decimal('0'),
        )
        await tracker.open_trade(
            inst_id='ETH-USDT', side='buy',
            size=Decimal('1'), entry_price=Decimal('3500'),
            fee=Decimal('0'),
        )

        assert tracker.open_trade_count == 2

    async def test_total_unrealized_pnl(self, tracker: PortfolioTracker):
        await tracker.open_trade(
            inst_id='BTC-USDT', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('67500'),
            fee=Decimal('0'),
        )

        pnl = tracker.total_unrealized_pnl({
            'BTC-USDT': Decimal('68000'),
        })
        # (68000 - 67500) * 0.01 = 5.0
        assert pnl == Decimal('5.00')


class TestPortfolioTimeLimits:
    async def test_no_expired_trades(self, tracker: PortfolioTracker):
        await tracker.open_trade(
            inst_id='BTC-USDT', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('67500'),
            fee=Decimal('0'),
        )

        expired = tracker.check_time_limits(time_limit_seconds=3600)
        assert expired == []


class TestPortfolioPersistence:
    async def test_initialize_loads_open_trades(
        self, mock_client: OKXClient, db: Database,
    ):
        # First session: open a trade
        t1 = PortfolioTracker(mock_client, db)
        await t1.initialize()
        await t1.open_trade(
            inst_id='BTC-USDT', side='buy',
            size=Decimal('0.01'), entry_price=Decimal('67500'),
            fee=Decimal('0'),
        )

        # Second session: should find the orphaned trade
        t2 = PortfolioTracker(mock_client, db)
        await t2.initialize()
        assert t2.open_trade_count == 1
        assert t2.has_open_trade('BTC-USDT')
