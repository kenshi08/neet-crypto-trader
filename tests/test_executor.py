"""Tests for the OrderExecutor — trade execution with triple barrier."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nct.config import BudgetConfig, OKXCredentials
from nct.db import Database
from nct.exchange.client import OKXClient
from nct.executor import OrderExecutor
from nct.portfolio.tracker import PortfolioTracker
from nct.risk.budget_manager import BudgetManager
from nct.risk.protections import ProtectionManager
from nct.risk.risk_manager import TradeDecision


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / 'test_executor.sqlite')
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def mock_client() -> OKXClient:
    client = OKXClient(OKXCredentials(demo_mode=True))
    client._market_api = MagicMock()
    client._trade_api = MagicMock()
    client._account_api = MagicMock()

    # Mock place_order to return a fill
    client._trade_api.place_order.return_value = {
        'code': '0',
        'data': [{
            'ordId': '12345',
            'clOrdId': '',
            'sCode': '0',
            'instId': 'BTC-USDT',
            'side': 'buy',
            'sz': '0.01',
            'avgPx': '67500',
            'fillSz': '0.01',
            'fee': '-0.675',
            'feeCcy': 'USDT',
            'state': 'filled',
        }],
    }

    # Mock algo orders
    client._trade_api.place_algo_order.return_value = {
        'code': '0',
        'data': [{'algoId': 'algo-123'}],
    }

    return client


@pytest.fixture
async def executor(mock_client: OKXClient, db: Database) -> OrderExecutor:
    budget_config = BudgetConfig(
        amount_usdt=Decimal('500'),
        max_position_pct=Decimal('20'),
    )
    budget_mgr = BudgetManager(budget_config, db)
    await budget_mgr.initialize()

    portfolio = PortfolioTracker(mock_client, db)
    await portfolio.initialize()

    protections = ProtectionManager()

    return OrderExecutor(
        client=mock_client,
        portfolio=portfolio,
        budget_manager=budget_mgr,
        protection_manager=protections,
    )


def _approved_decision() -> TradeDecision:
    return TradeDecision(
        approved=True,
        size=Decimal('0.01'),
        stop_loss_price=Decimal('65475'),
        take_profit_price=Decimal('70875'),
        time_limit_seconds=3600,
    )


class TestExecuteTrade:
    async def test_executes_approved_trade(self, executor: OrderExecutor):
        trade = await executor.execute_trade(
            inst_id='BTC-USDT',
            decision=_approved_decision(),
            strategy_name='momentum',
            signal_confidence=0.8,
        )

        assert trade is not None
        assert trade.inst_id == 'BTC-USDT'
        assert trade.trade_id > 0

    async def test_rejects_unapproved_decision(self, executor: OrderExecutor):
        decision = TradeDecision(approved=False, denial_reason='test')
        trade = await executor.execute_trade(
            inst_id='BTC-USDT', decision=decision,
        )
        assert trade is None

    async def test_places_stop_loss_algo_order(
        self, executor: OrderExecutor, mock_client: OKXClient,
    ):
        await executor.execute_trade(
            inst_id='BTC-USDT', decision=_approved_decision(),
        )

        # Check algo order was called (for SL and TP)
        assert mock_client._trade_api.place_algo_order.call_count >= 1

    async def test_updates_budget_on_execute(self, executor: OrderExecutor):
        await executor.execute_trade(
            inst_id='BTC-USDT', decision=_approved_decision(),
        )

        assert executor._budget.capital_deployed > 0

    async def test_tracks_in_portfolio(self, executor: OrderExecutor):
        await executor.execute_trade(
            inst_id='BTC-USDT', decision=_approved_decision(),
        )

        assert executor._portfolio.has_open_trade('BTC-USDT')


class TestAtomicStopLoss:
    """Safety invariant: no position may exist without a server-side stop-loss."""

    async def test_reverses_entry_when_stop_loss_fails(
        self, executor: OrderExecutor, mock_client: OKXClient,
    ):
        # Make place_algo_order raise to simulate SL placement failure
        mock_client._trade_api.place_algo_order.side_effect = Exception(
            'simulated SL placement failure',
        )

        trade = await executor.execute_trade(
            inst_id='BTC-USDT', decision=_approved_decision(),
        )

        # Execution must fail, no trade tracked, no budget consumed
        assert trade is None
        assert not executor._portfolio.has_open_trade('BTC-USDT')
        assert executor._budget.capital_deployed == Decimal(0)

        # The executor should have placed two dry-run orders:
        # 1. Entry BUY order
        # 2. Reversal SELL order (closing the unprotected entry)
        assert len(mock_client._dry_run_orders) == 2
        sides = [o.side.value for o in mock_client._dry_run_orders.values()]
        assert 'buy' in sides
        assert 'sell' in sides

    async def test_reverses_entry_when_take_profit_fails(
        self, executor: OrderExecutor, mock_client: OKXClient,
    ):
        # SL succeeds (first call), TP fails (second call)
        mock_client._trade_api.place_algo_order.side_effect = [
            {'code': '0', 'data': [{'algoId': 'sl-algo-1'}]},
            Exception('simulated TP placement failure'),
        ]

        trade = await executor.execute_trade(
            inst_id='BTC-USDT', decision=_approved_decision(),
        )

        assert trade is None
        assert not executor._portfolio.has_open_trade('BTC-USDT')
        assert executor._budget.capital_deployed == Decimal(0)
        # Entry + reversal = 2 dry-run orders
        assert len(mock_client._dry_run_orders) == 2


class TestCloseTrade:
    async def test_close_trade_records_pnl(self, executor: OrderExecutor):
        await executor.execute_trade(
            inst_id='BTC-USDT', decision=_approved_decision(),
        )

        pnl = await executor.close_trade(
            'BTC-USDT',
            current_price=Decimal('68000'),
            reason='take_profit',
        )

        assert pnl != 0
        assert not executor._portfolio.has_open_trade('BTC-USDT')

    async def test_close_stop_loss_updates_protections(
        self, executor: OrderExecutor,
    ):
        await executor.execute_trade(
            inst_id='BTC-USDT', decision=_approved_decision(),
        )

        await executor.close_trade(
            'BTC-USDT',
            current_price=Decimal('65000'),
            reason='stop_loss',
            was_stop_loss=True,
        )

        assert executor._budget.daily_stop_loss_count == 1


class TestExpiredTrades:
    async def test_check_and_close_expired(self, executor: OrderExecutor):
        await executor.execute_trade(
            inst_id='BTC-USDT', decision=_approved_decision(),
        )

        # Manually set opened_at to the past to simulate expiry
        from datetime import UTC, datetime, timedelta

        trade = executor._portfolio._open_trades['BTC-USDT']
        trade.opened_at = datetime.now(UTC) - timedelta(hours=2)

        closed = await executor.check_and_close_expired(
            time_limit_seconds=3600,
            current_prices={'BTC-USDT': Decimal('67800')},
        )

        assert 'BTC-USDT' in closed
        assert not executor._portfolio.has_open_trade('BTC-USDT')
