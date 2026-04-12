"""Tests for SL/TP barrier verification and repair."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from nct.exchange.models import OrderStatus, Side, Ticker
from nct.executor import OrderExecutor
from nct.portfolio.tracker import TrackedTrade


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.get_algo_order_status = AsyncMock(return_value=OrderStatus.PENDING)
    client.place_stop_loss = AsyncMock(return_value='new_sl_123')
    client.place_take_profit = AsyncMock(return_value='new_tp_456')
    client.get_ticker = AsyncMock(return_value=Ticker(
        inst_id='BTC-USD', last=Decimal('65000'), bid=Decimal('64990'),
        ask=Decimal('65010'), bid_size=Decimal('1'), ask_size=Decimal('1'),
        volume_24h=Decimal('1000'), timestamp=MagicMock(),
    ))
    return client


@pytest.fixture
def mock_portfolio():
    pt = MagicMock()
    pt.open_trades = {}
    pt.close_trade = AsyncMock(return_value=Decimal('0'))
    return pt


@pytest.fixture
def executor(mock_client, mock_portfolio):
    return OrderExecutor(
        client=mock_client,
        portfolio=mock_portfolio,
        budget_manager=MagicMock(),
        protection_manager=MagicMock(),
    )


def _make_trade(**overrides) -> TrackedTrade:
    defaults = {
        'trade_id': 1,
        'inst_id': 'BTC-USD',
        'side': 'buy',
        'size': Decimal('0.01'),
        'entry_price': Decimal('64000'),
        'fee': Decimal('0.1'),
        'stop_loss_algo_id': 'sl_abc',
        'take_profit_algo_id': 'tp_xyz',
        'stop_loss_price': Decimal('62000'),
        'take_profit_price': Decimal('67000'),
    }
    defaults.update(overrides)
    return TrackedTrade(**defaults)


class TestVerifyBarriersAllActive:
    @pytest.mark.asyncio
    async def test_no_action_when_barriers_active(self, executor, mock_portfolio, mock_client):
        trade = _make_trade()
        mock_portfolio.open_trades = {'BTC-USD': trade}
        mock_client.get_algo_order_status.return_value = OrderStatus.PENDING

        result = await executor.verify_barriers()
        assert result == []
        mock_client.place_stop_loss.assert_not_called()
        mock_client.place_take_profit.assert_not_called()


class TestVerifyBarriersMissingSL:
    @pytest.mark.asyncio
    async def test_sl_cancelled_is_repaired(self, executor, mock_portfolio, mock_client):
        trade = _make_trade()
        mock_portfolio.open_trades = {'BTC-USD': trade}

        # SL cancelled, TP still active
        async def _status(inst_id, algo_id):
            if algo_id == 'sl_abc':
                return OrderStatus.CANCELLED
            return OrderStatus.PENDING

        mock_client.get_algo_order_status.side_effect = _status

        await executor.verify_barriers()

        mock_client.place_stop_loss.assert_called_once_with(
            inst_id='BTC-USD',
            side=Side.SELL,
            size=Decimal('0.01'),
            trigger_price=Decimal('62000'),
        )
        assert trade.stop_loss_algo_id == 'new_sl_123'

    @pytest.mark.asyncio
    async def test_sl_repair_failure_attempts_close(self, executor, mock_portfolio, mock_client):
        trade = _make_trade()
        mock_portfolio.open_trades = {'BTC-USD': trade}

        mock_client.get_algo_order_status.return_value = OrderStatus.CANCELLED
        mock_client.place_stop_loss.side_effect = Exception('API error')

        await executor.verify_barriers()

        # Should have attempted to fetch price for emergency close
        mock_client.get_ticker.assert_called_with('BTC-USD')


class TestVerifyBarriersMissingTP:
    @pytest.mark.asyncio
    async def test_tp_cancelled_is_repaired(self, executor, mock_portfolio, mock_client):
        trade = _make_trade()
        mock_portfolio.open_trades = {'BTC-USD': trade}

        async def _status(inst_id, algo_id):
            if algo_id == 'tp_xyz':
                return OrderStatus.CANCELLED
            return OrderStatus.PENDING

        mock_client.get_algo_order_status.side_effect = _status

        await executor.verify_barriers()

        mock_client.place_take_profit.assert_called_once_with(
            inst_id='BTC-USD',
            side=Side.SELL,
            size=Decimal('0.01'),
            trigger_price=Decimal('67000'),
        )
        assert trade.take_profit_algo_id == 'new_tp_456'


class TestVerifyBarriersNoAlgoId:
    @pytest.mark.asyncio
    async def test_no_check_if_no_algo_id(self, executor, mock_portfolio, mock_client):
        trade = _make_trade(stop_loss_algo_id='', take_profit_algo_id='')
        mock_portfolio.open_trades = {'BTC-USD': trade}

        await executor.verify_barriers()

        mock_client.get_algo_order_status.assert_not_called()
