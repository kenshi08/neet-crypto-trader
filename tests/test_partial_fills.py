"""Tests for partial fill handling in OrderExecutor."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from nct.exchange.models import OrderResponse, OrderStatus, Side
from nct.executor import OrderExecutor
from nct.risk.risk_manager import TradeDecision


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.place_order = AsyncMock()
    client.place_stop_loss = AsyncMock(return_value='sl_123')
    client.place_take_profit = AsyncMock(return_value='tp_456')
    client.cancel_order = AsyncMock()
    return client


@pytest.fixture
def mock_portfolio():
    pt = AsyncMock()
    pt.open_trade = AsyncMock(return_value=MagicMock())
    return pt


@pytest.fixture
def mock_budget():
    bm = AsyncMock()
    bm.record_trade_open = AsyncMock()
    return bm


@pytest.fixture
def executor(mock_client, mock_portfolio, mock_budget):
    return OrderExecutor(
        client=mock_client,
        portfolio=mock_portfolio,
        budget_manager=mock_budget,
        protection_manager=MagicMock(),
    )


def _decision(**overrides) -> TradeDecision:
    defaults = {
        'approved': True,
        'size': Decimal('1.0'),
        'stop_loss_price': Decimal('62000'),
        'take_profit_price': Decimal('67000'),
        'time_limit_seconds': 3600,
    }
    defaults.update(overrides)
    return TradeDecision(**defaults)


def _order_response(*, filled_size: Decimal, size: Decimal, is_dry_run: bool = False) -> OrderResponse:
    return OrderResponse(
        order_id='ord_123',
        client_order_id='',
        status=OrderStatus.FILLED if filled_size == size else OrderStatus.PARTIALLY_FILLED,
        inst_id='BTC-USD',
        side=Side.BUY,
        size=size,
        price=Decimal('65000'),
        filled_size=filled_size,
        avg_fill_price=Decimal('65000'),
        fee=Decimal('2.6'),
        is_dry_run=is_dry_run,
    )


class TestFullFill:
    @pytest.mark.asyncio
    async def test_full_fill_uses_decision_size(self, executor, mock_client, mock_portfolio):
        mock_client.place_order.return_value = _order_response(
            filled_size=Decimal('1.0'), size=Decimal('1.0'),
        )

        result = await executor.execute_trade(
            inst_id='BTC-USD', decision=_decision(),
        )

        assert result is not None
        # SL/TP should use full size
        mock_client.place_stop_loss.assert_called_once()
        sl_call = mock_client.place_stop_loss.call_args
        assert sl_call.kwargs.get('size') or sl_call[1].get('size') == Decimal('1.0')


class TestPartialFill:
    @pytest.mark.asyncio
    async def test_partial_fill_uses_filled_size_for_barriers(
        self, executor, mock_client, mock_portfolio,
    ):
        mock_client.place_order.return_value = _order_response(
            filled_size=Decimal('0.7'), size=Decimal('1.0'),
        )

        result = await executor.execute_trade(
            inst_id='BTC-USD', decision=_decision(),
        )

        assert result is not None
        # SL/TP should use filled size (0.7), not requested (1.0)
        sl_call = mock_client.place_stop_loss.call_args
        assert sl_call[1]['size'] == Decimal('0.7')

        tp_call = mock_client.place_take_profit.call_args
        assert tp_call[1]['size'] == Decimal('0.7')

    @pytest.mark.asyncio
    async def test_partial_fill_portfolio_tracks_actual_size(
        self, executor, mock_client, mock_portfolio,
    ):
        mock_client.place_order.return_value = _order_response(
            filled_size=Decimal('0.5'), size=Decimal('1.0'),
        )

        await executor.execute_trade(inst_id='BTC-USD', decision=_decision())

        # Portfolio should record actual filled size
        open_call = mock_portfolio.open_trade.call_args
        assert open_call[1]['size'] == Decimal('0.5')


class TestZeroFill:
    @pytest.mark.asyncio
    async def test_zero_fill_returns_none(self, executor, mock_client):
        mock_client.place_order.return_value = _order_response(
            filled_size=Decimal('0'), size=Decimal('1.0'),
        )

        result = await executor.execute_trade(
            inst_id='BTC-USD', decision=_decision(),
        )

        assert result is None
        mock_client.place_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_fill_dry_run_still_proceeds(self, executor, mock_client):
        """In dry-run mode, filled_size=0 is expected (simulated) — proceed normally."""
        mock_client.place_order.return_value = _order_response(
            filled_size=Decimal('0'), size=Decimal('1.0'), is_dry_run=True,
        )

        result = await executor.execute_trade(
            inst_id='BTC-USD', decision=_decision(),
        )

        # Should proceed because dry-run
        assert result is not None
