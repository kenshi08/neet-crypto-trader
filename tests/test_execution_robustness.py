"""Tests for Phase 1 execution robustness fixes.

Covers:
- Bybit get_algo_order_status() fix (#75)
- Coinbase get_algo_order_status() fix (#76)
- OKX get_algo_order_status() fix (same class of bug)
- Idempotency key generation on retries (#77)
- Barrier repair backoff (#78)
"""

from __future__ import annotations

import time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nct.config import BybitCredentials, CoinbaseCredentials
from nct.exchange.bybit_client import BybitClient
from nct.exchange.coinbase_client import CoinbaseClient
from nct.exchange.models import (
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
    Ticker,
)
from nct.executor import (
    BARRIER_REPAIR_BACKOFF_BASE,
    BARRIER_REPAIR_MAX_ATTEMPTS,
    OrderExecutor,
    RepairState,
)
from nct.portfolio.tracker import TrackedTrade


# ===================================================================
# Helpers
# ===================================================================


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


def _mock_ticker():
    return Ticker(
        inst_id='BTC-USD', last=Decimal('65000'), bid=Decimal('64990'),
        ask=Decimal('65010'), bid_size=Decimal('1'), ask_size=Decimal('1'),
        volume_24h=Decimal('1000'), timestamp=MagicMock(),
    )


def _make_executor(mock_client=None, mock_portfolio=None):
    client = mock_client or AsyncMock()
    portfolio = mock_portfolio or MagicMock()
    portfolio.open_trades = {}
    portfolio.close_trade = AsyncMock(return_value=Decimal('0'))
    return OrderExecutor(
        client=client,
        portfolio=portfolio,
        budget_manager=MagicMock(),
        protection_manager=MagicMock(),
    )


# ===================================================================
# Issue #75 — Bybit get_algo_order_status()
# ===================================================================


class TestBybitAlgoOrderStatus:
    def _mock_bybit(self) -> BybitClient:
        client = BybitClient(BybitCredentials(demo_mode=True))
        client._session = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_dry_run_returns_pending(self):
        client = self._mock_bybit()
        status = await client.get_algo_order_status('BTC-USDT', 'dry_sl_abc')
        assert status == OrderStatus.PENDING
        client._session.get_open_orders.assert_not_called()

    @pytest.mark.asyncio
    async def test_active_order_returns_pending(self):
        client = self._mock_bybit()
        client._session.get_open_orders.return_value = {
            'retCode': 0,
            'result': {'list': [{'orderStatus': 'Untriggered'}]},
        }
        status = await client.get_algo_order_status('BTC-USDT', 'order-123')
        assert status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_filled_order(self):
        client = self._mock_bybit()
        client._session.get_open_orders.return_value = {
            'retCode': 0,
            'result': {'list': [{'orderStatus': 'Filled'}]},
        }
        status = await client.get_algo_order_status('BTC-USDT', 'order-123')
        assert status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_empty_list_returns_cancelled(self):
        client = self._mock_bybit()
        client._session.get_open_orders.return_value = {
            'retCode': 0,
            'result': {'list': []},
        }
        status = await client.get_algo_order_status('BTC-USDT', 'order-123')
        assert status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_exception_returns_cancelled(self):
        client = self._mock_bybit()
        client._session.get_open_orders.side_effect = Exception('network error')
        status = await client.get_algo_order_status('BTC-USDT', 'order-123')
        assert status == OrderStatus.CANCELLED


# ===================================================================
# Issue #76 — Coinbase get_algo_order_status()
# ===================================================================


class TestCoinbaseAlgoOrderStatus:
    def _mock_coinbase(self) -> CoinbaseClient:
        client = CoinbaseClient(CoinbaseCredentials(
            api_key='test-key', api_secret='test-secret', demo_mode=False,
        ))
        client._session = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_dry_run_returns_pending(self):
        client = self._mock_coinbase()
        status = await client.get_algo_order_status('BTC-USD', 'dry_sl_abc')
        assert status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_open_order_returns_pending(self):
        client = self._mock_coinbase()
        client._session.get_order.return_value = SimpleNamespace(
            order=SimpleNamespace(status='OPEN'),
        )
        status = await client.get_algo_order_status('BTC-USD', 'order-123')
        assert status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_filled_order(self):
        client = self._mock_coinbase()
        client._session.get_order.return_value = SimpleNamespace(
            order=SimpleNamespace(status='FILLED'),
        )
        status = await client.get_algo_order_status('BTC-USD', 'order-123')
        assert status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_cancelled_order(self):
        client = self._mock_coinbase()
        client._session.get_order.return_value = SimpleNamespace(
            order=SimpleNamespace(status='CANCELLED'),
        )
        status = await client.get_algo_order_status('BTC-USD', 'order-123')
        assert status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_exception_returns_cancelled(self):
        client = self._mock_coinbase()
        client._session.get_order.side_effect = Exception('API error')
        status = await client.get_algo_order_status('BTC-USD', 'order-123')
        assert status == OrderStatus.CANCELLED


# ===================================================================
# Issue #77 — Idempotency key generation
# ===================================================================


class TestIdempotencyKeyGeneration:
    @pytest.mark.asyncio
    async def test_bybit_generates_client_order_id_if_empty(self):
        client = BybitClient(BybitCredentials(api_key='k', api_secret='s', demo_mode=True))
        client._session = MagicMock()
        client._session.place_order.return_value = {
            'retCode': 0,
            'result': {'orderId': 'order-1', 'orderLinkId': ''},
        }

        order = OrderRequest(
            inst_id='BTC-USDT', side=Side.BUY,
            order_type=OrderType.MARKET, size=Decimal('0.01'),
        )
        assert order.client_order_id == ''

        await client.place_order(order)

        # After the call, client_order_id should be populated
        assert order.client_order_id.startswith('nct_')
        # And it should have been passed to the exchange
        call_kwargs = client._session.place_order.call_args.kwargs
        assert call_kwargs['orderLinkId'] == order.client_order_id

    @pytest.mark.asyncio
    async def test_bybit_preserves_existing_client_order_id(self):
        client = BybitClient(BybitCredentials(api_key='k', api_secret='s', demo_mode=True))
        client._session = MagicMock()
        client._session.place_order.return_value = {
            'retCode': 0,
            'result': {'orderId': 'order-1', 'orderLinkId': 'my-id'},
        }

        order = OrderRequest(
            inst_id='BTC-USDT', side=Side.BUY,
            order_type=OrderType.MARKET, size=Decimal('0.01'),
            client_order_id='my-id',
        )
        await client.place_order(order)

        assert order.client_order_id == 'my-id'
        call_kwargs = client._session.place_order.call_args.kwargs
        assert call_kwargs['orderLinkId'] == 'my-id'


# ===================================================================
# Issue #78 — Barrier repair backoff
# ===================================================================


class TestRepairState:
    def test_first_attempt_always_allowed(self):
        state = RepairState()
        assert state.should_attempt(time.monotonic()) is True

    def test_second_attempt_requires_backoff(self):
        state = RepairState()
        now = time.monotonic()
        state.record_attempt(now)

        # Immediately after first attempt — should be blocked
        assert state.should_attempt(now + 1) is False

        # After base backoff (60s) — should be allowed
        assert state.should_attempt(now + BARRIER_REPAIR_BACKOFF_BASE + 1) is True

    def test_exponential_increase(self):
        state = RepairState()
        now = time.monotonic()

        # First attempt at t=0
        state.record_attempt(now)
        # Backoff: 60s

        # Second attempt at t=61
        state.record_attempt(now + 61)
        # Backoff: 120s

        # At t=130 (69s after second) — still blocked (need 120s)
        assert state.should_attempt(now + 130) is False

        # At t=182 (121s after second) — allowed
        assert state.should_attempt(now + 182) is True

    def test_max_attempts_blocks_forever(self):
        state = RepairState()
        now = time.monotonic()
        for i in range(BARRIER_REPAIR_MAX_ATTEMPTS):
            state.record_attempt(now + i * 1000)

        assert state.should_attempt(now + 100000) is False


class TestBarrierRepairBackoff:
    @pytest.mark.asyncio
    async def test_first_repair_proceeds_immediately(self):
        mock_client = AsyncMock()
        mock_client.get_algo_order_status = AsyncMock(return_value=OrderStatus.CANCELLED)
        mock_client.place_stop_loss = AsyncMock(return_value='new_sl')
        mock_client.place_take_profit = AsyncMock(return_value='new_tp')

        executor = _make_executor(mock_client=mock_client)
        trade = _make_trade()
        executor._portfolio.open_trades = {'BTC-USD': trade}

        await executor.verify_barriers()

        # Should have attempted repair on first call
        mock_client.place_stop_loss.assert_called_once()
        mock_client.place_take_profit.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_repair_blocked_by_backoff(self):
        mock_client = AsyncMock()
        mock_client.get_algo_order_status = AsyncMock(return_value=OrderStatus.CANCELLED)
        mock_client.place_stop_loss = AsyncMock(return_value='new_sl')
        mock_client.place_take_profit = AsyncMock(return_value='new_tp')

        executor = _make_executor(mock_client=mock_client)
        trade = _make_trade()
        executor._portfolio.open_trades = {'BTC-USD': trade}

        # First call — repairs
        await executor.verify_barriers()
        assert mock_client.place_stop_loss.call_count == 1

        # Reset the algo IDs back to cancelled state (repair returned new IDs)
        trade.stop_loss_algo_id = 'sl_still_bad'
        trade.take_profit_algo_id = 'tp_still_bad'

        # Second call immediately — blocked by backoff
        await executor.verify_barriers()
        assert mock_client.place_stop_loss.call_count == 1  # no new call

    @pytest.mark.asyncio
    async def test_healthy_barriers_reset_repair_state(self):
        mock_client = AsyncMock()
        mock_client.place_stop_loss = AsyncMock(return_value='new_sl')
        mock_client.place_take_profit = AsyncMock(return_value='new_tp')

        executor = _make_executor(mock_client=mock_client)
        trade = _make_trade()
        executor._portfolio.open_trades = {'BTC-USD': trade}

        # First call with cancelled barriers
        mock_client.get_algo_order_status = AsyncMock(return_value=OrderStatus.CANCELLED)
        await executor.verify_barriers()
        assert 'BTC-USD' in executor._repair_states

        # Second call with healthy barriers
        mock_client.get_algo_order_status = AsyncMock(return_value=OrderStatus.PENDING)
        await executor.verify_barriers()
        assert 'BTC-USD' not in executor._repair_states

    @pytest.mark.asyncio
    async def test_max_attempts_closes_position(self):
        mock_client = AsyncMock()
        mock_client.get_algo_order_status = AsyncMock(return_value=OrderStatus.CANCELLED)
        mock_client.place_stop_loss = AsyncMock(return_value='new_sl')
        mock_client.place_take_profit = AsyncMock(return_value='new_tp')
        mock_client.get_ticker = AsyncMock(return_value=_mock_ticker())

        executor = _make_executor(mock_client=mock_client)
        trade = _make_trade()
        executor._portfolio.open_trades = {'BTC-USD': trade}

        # Simulate max attempts already reached
        executor._repair_states['BTC-USD'] = RepairState(
            attempt_count=BARRIER_REPAIR_MAX_ATTEMPTS,
            last_attempt_time=time.monotonic(),
        )

        await executor.verify_barriers()

        # Should have attempted to close the position
        executor._portfolio.close_trade.assert_called_once()
