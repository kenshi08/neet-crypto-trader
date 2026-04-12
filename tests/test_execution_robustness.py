"""Tests for execution robustness fixes.

Covers:
- Bybit get_algo_order_status() fix (#75)
- Coinbase get_algo_order_status() fix (#76)
- OKX get_algo_order_status() fix (same class of bug)
- Idempotency key generation on retries (#77)
- Barrier repair backoff (#78)
- Fill-price re-query on missing avg_fill_price (#87)
"""

from __future__ import annotations

import time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nct.config import BybitCredentials, CoinbaseCredentials
from nct.exchange.bybit_client import BybitClient
from nct.exchange.coinbase_client import CoinbaseClient
from nct.exchange.models import (
    OrderRequest,
    OrderResponse,
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
from nct.risk.risk_manager import TradeDecision

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
    portfolio.open_trade = AsyncMock(return_value=_make_trade())
    portfolio.close_trade = AsyncMock(return_value=Decimal('0'))
    budget_mgr = MagicMock()
    budget_mgr.record_trade_open = AsyncMock()
    return OrderExecutor(
        client=client,
        portfolio=portfolio,
        budget_manager=budget_mgr,
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


# ===================================================================
# Issue #87 — Fill-price re-query on missing avg_fill_price
# ===================================================================


class TestFillPriceRequery:
    """Verify that the executor re-queries the exchange when avg_fill_price
    is missing, and reverses the entry when the price cannot be determined."""

    def _make_decision(self) -> TradeDecision:
        return TradeDecision(
            approved=True,
            size=Decimal('0.01'),
            stop_loss_price=Decimal('62000'),
            take_profit_price=Decimal('67000'),
            time_limit_seconds=3600,
        )

    @pytest.mark.asyncio
    async def test_uses_fill_price_from_initial_response(self):
        """When avg_fill_price is present, no re-query needed."""
        client = AsyncMock()
        client.place_order.return_value = OrderResponse(
            order_id='ord_1', client_order_id='', status=OrderStatus.FILLED,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, filled_size=Decimal('0.01'),
            avg_fill_price=Decimal('64500'), fee=Decimal('0.1'),
        )
        client.place_stop_loss.return_value = 'sl_1'
        client.place_take_profit.return_value = 'tp_1'

        executor = _make_executor(mock_client=client)
        trade = await executor.execute_trade(
            inst_id='BTC-USD', decision=self._make_decision(),
        )

        assert trade is not None
        # Verify open_trade was called with the correct entry price
        call_kwargs = executor._portfolio.open_trade.call_args
        assert call_kwargs.kwargs.get('entry_price') == Decimal('64500')
        client.get_order_detail.assert_not_called()

    @pytest.mark.asyncio
    async def test_requery_recovers_fill_price(self):
        """When initial response has no fill price, re-query returns it."""
        from nct.exchange.models import OrderResponse
        # Initial response: no fill price
        client = AsyncMock()
        client.place_order.return_value = OrderResponse(
            order_id='ord_2', client_order_id='', status=OrderStatus.FILLED,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, filled_size=Decimal('0.01'),
            avg_fill_price=None, fee=Decimal('0.1'),
            is_dry_run=False,
        )
        # Re-query returns fill price
        client.get_order_detail.return_value = OrderResponse(
            order_id='ord_2', client_order_id='', status=OrderStatus.FILLED,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, filled_size=Decimal('0.01'),
            avg_fill_price=Decimal('64200'), fee=Decimal('0.1'),
        )
        client.place_stop_loss.return_value = 'sl_2'
        client.place_take_profit.return_value = 'tp_2'

        executor = _make_executor(mock_client=client)
        trade = await executor.execute_trade(
            inst_id='BTC-USD', decision=self._make_decision(),
        )

        assert trade is not None
        call_kwargs = executor._portfolio.open_trade.call_args
        assert call_kwargs.kwargs.get('entry_price') == Decimal('64200')
        client.get_order_detail.assert_called()

    @pytest.mark.asyncio
    async def test_reverses_entry_when_fill_price_never_found(self):
        """When fill price is missing and re-query fails, entry is reversed."""
        client = AsyncMock()
        client.place_order.return_value = OrderResponse(
            order_id='ord_3', client_order_id='', status=OrderStatus.FILLED,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, filled_size=Decimal('0.01'),
            avg_fill_price=None, fee=Decimal('0.1'),
            is_dry_run=False,
        )
        # Re-query never returns a price
        client.get_order_detail.return_value = OrderResponse(
            order_id='ord_3', client_order_id='', status=OrderStatus.FILLED,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, filled_size=Decimal('0.01'),
            avg_fill_price=None, fee=Decimal('0'),
        )

        executor = _make_executor(mock_client=client)
        trade = await executor.execute_trade(
            inst_id='BTC-USD', decision=self._make_decision(),
        )

        # Trade should be None — entry was reversed
        assert trade is None
        # Reversal order should have been placed (second place_order call)
        assert client.place_order.call_count == 2
        reversal_order = client.place_order.call_args_list[1][0][0]
        assert reversal_order.side == Side.SELL

    @pytest.mark.asyncio
    async def test_dry_run_uses_sl_tp_midpoint(self):
        """In dry-run mode, derive entry price from SL/TP midpoint (simulated)."""
        client = AsyncMock()
        client.place_order.return_value = OrderResponse(
            order_id='dry_123', client_order_id='', status=OrderStatus.FILLED,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, filled_size=Decimal('0.01'),
            avg_fill_price=None, fee=Decimal('0'),
            is_dry_run=True,
        )
        client.place_stop_loss.return_value = 'sl_dry'
        client.place_take_profit.return_value = 'tp_dry'

        executor = _make_executor(mock_client=client)
        decision = self._make_decision()
        trade = await executor.execute_trade(
            inst_id='BTC-USD', decision=decision,
        )

        assert trade is not None
        # Midpoint of SL(62000) and TP(67000) = 64500
        call_kwargs = executor._portfolio.open_trade.call_args
        assert call_kwargs.kwargs.get('entry_price') == Decimal('64500')
        client.get_order_detail.assert_not_called()

    @pytest.mark.asyncio
    async def test_requery_retries_up_to_max_attempts(self):
        """Re-query is called up to max_attempts before giving up."""
        client = AsyncMock()
        client.place_order.return_value = OrderResponse(
            order_id='ord_4', client_order_id='', status=OrderStatus.FILLED,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, filled_size=Decimal('0.01'),
            avg_fill_price=None, fee=Decimal('0.1'),
            is_dry_run=False,
        )
        # All re-queries return no fill price
        client.get_order_detail.return_value = OrderResponse(
            order_id='ord_4', client_order_id='', status=OrderStatus.PENDING,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, avg_fill_price=None,
        )

        executor = _make_executor(mock_client=client)
        trade = await executor.execute_trade(
            inst_id='BTC-USD', decision=self._make_decision(),
        )

        assert trade is None
        assert client.get_order_detail.call_count == 3  # default max_attempts

    @pytest.mark.asyncio
    async def test_requery_recovers_on_second_attempt(self):
        """Re-query finds the price on the second attempt."""
        client = AsyncMock()
        client.place_order.return_value = OrderResponse(
            order_id='ord_5', client_order_id='', status=OrderStatus.FILLED,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, filled_size=Decimal('0.01'),
            avg_fill_price=None, fee=Decimal('0.1'),
            is_dry_run=False,
        )
        # First re-query: no price; second: has price
        no_price = OrderResponse(
            order_id='ord_5', client_order_id='', status=OrderStatus.PENDING,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, avg_fill_price=None,
        )
        has_price = OrderResponse(
            order_id='ord_5', client_order_id='', status=OrderStatus.FILLED,
            inst_id='BTC-USD', side=Side.BUY, size=Decimal('0.01'),
            price=None, filled_size=Decimal('0.01'),
            avg_fill_price=Decimal('63800'), fee=Decimal('0.1'),
        )
        client.get_order_detail.side_effect = [no_price, has_price]
        client.place_stop_loss.return_value = 'sl_5'
        client.place_take_profit.return_value = 'tp_5'

        executor = _make_executor(mock_client=client)
        trade = await executor.execute_trade(
            inst_id='BTC-USD', decision=self._make_decision(),
        )

        assert trade is not None
        call_kwargs = executor._portfolio.open_trade.call_args
        assert call_kwargs.kwargs.get('entry_price') == Decimal('63800')
        assert client.get_order_detail.call_count == 2
