"""Tests for the OKX exchange client."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nct.config import OKXCredentials
from nct.exchange.client import (
    AuthenticationError,
    ExchangeError,
    OKXClient,
    OrderError,
    RateLimitError,
    retrier,
)
from nct.exchange.models import OrderRequest, OrderType, Side, TdMode

# ---------------------------------------------------------------------------
# Retrier tests
# ---------------------------------------------------------------------------


class TestRetrier:
    async def test_succeeds_on_first_try(self):
        call_count = 0

        @retrier
        async def succeeds():
            nonlocal call_count
            call_count += 1
            return 'ok'

        result = await succeeds()
        assert result == 'ok'
        assert call_count == 1

    async def test_retries_on_exchange_error(self):
        call_count = 0

        @retrier
        async def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ExchangeError('transient')
            return 'ok'

        with patch('nct.exchange.client.asyncio.sleep', new_callable=AsyncMock):
            result = await fails_then_succeeds()

        assert result == 'ok'
        assert call_count == 3

    async def test_does_not_retry_auth_error(self):
        call_count = 0

        @retrier
        async def auth_fails():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError('bad key')

        with pytest.raises(AuthenticationError):
            await auth_fails()
        assert call_count == 1

    async def test_exhausts_retries(self):
        @retrier
        async def always_fails():
            raise ExchangeError('permanent')

        with (
            patch('nct.exchange.client.asyncio.sleep', new_callable=AsyncMock),
            pytest.raises(ExchangeError, match='failed after 4 retries'),
        ):
            await always_fails()

    async def test_retries_on_connection_error(self):
        call_count = 0

        @retrier
        async def network_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError('network down')
            return 'recovered'

        with patch('nct.exchange.client.asyncio.sleep', new_callable=AsyncMock):
            result = await network_error()

        assert result == 'recovered'
        assert call_count == 2


# ---------------------------------------------------------------------------
# OKXClient response checking
# ---------------------------------------------------------------------------


class TestCheckResponse:
    def setup_method(self):
        self.client = OKXClient(OKXCredentials(demo_mode=True))

    def test_success_returns_data(self):
        result = {'code': '0', 'data': [{'instId': 'BTC-USDT'}]}
        data = self.client._check_response(result)
        assert data == [{'instId': 'BTC-USDT'}]

    def test_auth_error(self):
        result = {'code': '50001', 'msg': 'Invalid signature'}
        with pytest.raises(AuthenticationError):
            self.client._check_response(result)

    def test_rate_limit_error(self):
        result = {'code': '50011', 'msg': 'Too many requests'}
        with pytest.raises(RateLimitError):
            self.client._check_response(result)

    def test_order_error(self):
        result = {'code': '51004', 'msg': 'Insufficient balance'}
        with pytest.raises(OrderError):
            self.client._check_response(result)

    def test_generic_error(self):
        result = {'code': '99999', 'msg': 'Unknown'}
        with pytest.raises(ExchangeError):
            self.client._check_response(result)

    def test_empty_data(self):
        result = {'code': '0'}
        data = self.client._check_response(result)
        assert data == []


# ---------------------------------------------------------------------------
# Dry-run order simulation
# ---------------------------------------------------------------------------


class TestDryRunOrders:
    def setup_method(self):
        self.client = OKXClient(OKXCredentials(demo_mode=True))

    def test_creates_dry_run_order(self):
        order = OrderRequest(
            inst_id='BTC-USDT',
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal('0.01'),
            price=Decimal('67500'),
            td_mode=TdMode.CASH,
        )
        response = self.client._create_dry_run_order(order)

        assert response.order_id.startswith('dry_')
        assert response.is_dry_run is True
        assert response.inst_id == 'BTC-USDT'
        assert response.side == Side.BUY
        assert response.size == Decimal('0.01')
        assert response.filled_size == Decimal('0.01')
        assert response.fee > 0  # 0.1% fee simulated

    def test_dry_run_order_tracked(self):
        order = OrderRequest(
            inst_id='ETH-USDT',
            side=Side.SELL,
            order_type=OrderType.MARKET,
            size=Decimal('1.0'),
        )
        response = self.client._create_dry_run_order(order)
        assert response.order_id in self.client._dry_run_orders

    def test_dry_run_market_order_no_price(self):
        order = OrderRequest(
            inst_id='SOL-USDT',
            side=Side.BUY,
            order_type=OrderType.MARKET,
            size=Decimal('10'),
        )
        response = self.client._create_dry_run_order(order)
        assert response.price is None

    async def test_cancel_dry_run_order(self):
        order = OrderRequest(
            inst_id='BTC-USDT',
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal('0.01'),
            price=Decimal('65000'),
        )
        response = self.client._create_dry_run_order(order)
        assert response.order_id in self.client._dry_run_orders

        await self.client.cancel_order('BTC-USDT', response.order_id)
        assert response.order_id not in self.client._dry_run_orders


# ---------------------------------------------------------------------------
# Client properties
# ---------------------------------------------------------------------------


class TestClientProperties:
    def test_demo_mode_property(self):
        client = OKXClient(OKXCredentials(demo_mode=True))
        assert client.is_demo is True

    def test_live_mode_property(self):
        client = OKXClient(OKXCredentials(demo_mode=False))
        assert client.is_demo is False


# ---------------------------------------------------------------------------
# Integration-style tests (mocked SDK)
# ---------------------------------------------------------------------------


class TestClientWithMockedSDK:
    def _mock_client(self) -> OKXClient:
        """Create an OKXClient with mocked SDK instances to prevent real API calls."""
        creds = OKXCredentials(
            api_key='test-key',
            api_secret='test-secret',
            passphrase='test-pass',
            demo_mode=True,
        )
        client = OKXClient(creds)
        # Pre-set SDK attrs so _ensure_sdk() skips real initialization
        client._market_api = MagicMock()
        client._trade_api = MagicMock()
        client._account_api = MagicMock()
        return client

    async def test_get_ticker(self):
        client = self._mock_client()
        client._market_api.get_ticker.return_value = {
            'code': '0',
            'data': [{
                'instId': 'BTC-USDT',
                'last': '67500.5',
                'bidPx': '67500.0',
                'askPx': '67501.0',
                'bidSz': '1.5',
                'askSz': '2.0',
                'vol24h': '12345.67',
                'ts': '1712000000000',
            }],
        }

        ticker = await client.get_ticker('BTC-USDT')

        assert ticker.inst_id == 'BTC-USDT'
        assert ticker.last == Decimal('67500.5')
        client._market_api.get_ticker.assert_called_once_with(instId='BTC-USDT')

    async def test_get_candlesticks(self):
        client = self._mock_client()
        client._market_api.get_candlesticks.return_value = {
            'code': '0',
            'data': [
                ['1712000000000', '67000', '67800', '66900', '67500', '1234', '0', '0', '1'],
                ['1711996400000', '66800', '67100', '66700', '67000', '987', '0', '0', '1'],
            ],
        }

        candles = await client.get_candlesticks('BTC-USDT', bar='15m', limit=2)

        assert len(candles) == 2
        # Should be sorted by timestamp ascending
        assert candles[0].timestamp < candles[1].timestamp

    async def test_get_balance(self):
        client = self._mock_client()
        client._account_api.get_account_balance.return_value = {
            'code': '0',
            'data': [{
                'details': [
                    {'ccy': 'USDT', 'bal': '10000', 'availBal': '8500', 'frozenBal': '1500'},
                    {'ccy': 'BTC', 'bal': '0.5', 'availBal': '0.5', 'frozenBal': '0'},
                ],
            }],
        }

        balances = await client.get_balance()

        assert len(balances) == 2
        usdt = next(b for b in balances if b.currency == 'USDT')
        assert usdt.total == Decimal('10000')
        assert usdt.available == Decimal('8500')

    async def test_place_order_live(self):
        client = self._mock_client()
        client._trade_api.place_order.return_value = {
            'code': '0',
            'data': [{
                'ordId': '111',
                'clOrdId': '',
                'sCode': '0',
                'instId': 'BTC-USDT',
                'side': 'buy',
                'sz': '0.01',
            }],
        }

        order = OrderRequest(
            inst_id='BTC-USDT',
            side=Side.BUY,
            order_type=OrderType.MARKET,
            size=Decimal('0.01'),
        )
        response = await client.place_order(order)

        assert response.order_id == '111'
        client._trade_api.place_order.assert_called_once()

    async def test_validate_connection_success(self):
        client = self._mock_client()
        client._account_api.get_account_balance.return_value = {
            'code': '0',
            'data': [{'details': [{'ccy': 'USDT', 'bal': '1000', 'availBal': '1000'}]}],
        }

        result = await client.validate_connection()
        assert result is True

    async def test_validate_connection_auth_failure(self):
        client = self._mock_client()
        client._account_api.get_account_balance.return_value = {
            'code': '50001',
            'msg': 'Invalid signature',
        }

        result = await client.validate_connection()
        assert result is False
