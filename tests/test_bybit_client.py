"""Tests for the BybitClient — mocked pybit SDK responses."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from nct.config import BybitCredentials
from nct.exceptions import AuthenticationError, ExchangeError, OrderError, RateLimitError
from nct.exchange.bybit_client import (
    BybitClient,
    _to_bybit_symbol,
    _to_bybit_timeframe,
    _to_inst_id,
)
from nct.exchange.models import OrderRequest, OrderType, Side

# ---------------------------------------------------------------------------
# Symbol / timeframe helpers
# ---------------------------------------------------------------------------


class TestSymbolConversion:
    def test_to_bybit_symbol(self):
        assert _to_bybit_symbol('BTC-USDT') == 'BTCUSDT'
        assert _to_bybit_symbol('ETH-USDC') == 'ETHUSDC'
        assert _to_bybit_symbol('SOL-USDT') == 'SOLUSDT'

    def test_to_inst_id(self):
        assert _to_inst_id('BTCUSDT') == 'BTC-USDT'
        assert _to_inst_id('ETHUSDC') == 'ETH-USDC'
        assert _to_inst_id('SOLUSD') == 'SOL-USD'

    def test_to_bybit_timeframe(self):
        assert _to_bybit_timeframe('1m') == '1'
        assert _to_bybit_timeframe('15m') == '15'
        assert _to_bybit_timeframe('1H') == '60'
        assert _to_bybit_timeframe('4H') == '240'
        assert _to_bybit_timeframe('1D') == 'D'


# ---------------------------------------------------------------------------
# Response checking
# ---------------------------------------------------------------------------


class TestCheckResponse:
    def setup_method(self):
        self.client = BybitClient(BybitCredentials(demo_mode=True))

    def test_success_returns_result(self):
        response = {'retCode': 0, 'retMsg': 'OK', 'result': {'list': [1, 2]}}
        data = self.client._check_response(response)
        assert data == {'list': [1, 2]}

    def test_auth_error(self):
        response = {'retCode': 10003, 'retMsg': 'Invalid API key'}
        with pytest.raises(AuthenticationError):
            self.client._check_response(response)

    def test_rate_limit_error(self):
        response = {'retCode': 10018, 'retMsg': 'rate limit exceeded'}
        with pytest.raises(RateLimitError):
            self.client._check_response(response)

    def test_order_error(self):
        response = {'retCode': 110001, 'retMsg': 'Order failed'}
        with pytest.raises(OrderError):
            self.client._check_response(response)

    def test_generic_error(self):
        response = {'retCode': 99999, 'retMsg': 'Unknown error'}
        with pytest.raises(ExchangeError):
            self.client._check_response(response)

    def test_non_dict_response(self):
        with pytest.raises(ExchangeError):
            self.client._check_response('not a dict')


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestClientProperties:
    def test_demo_mode(self):
        client = BybitClient(BybitCredentials(demo_mode=True))
        assert client.is_demo is True

    def test_live_mode(self):
        client = BybitClient(BybitCredentials(demo_mode=False))
        assert client.is_demo is False


# ---------------------------------------------------------------------------
# Mocked SDK integration tests
# ---------------------------------------------------------------------------


class TestBybitClientWithMockedSDK:
    def _mock_client(self) -> BybitClient:
        creds = BybitCredentials(
            api_key='test-key',
            api_secret='test-secret',
            demo_mode=True,
        )
        client = BybitClient(creds)
        client._session = MagicMock()
        return client

    async def test_get_ticker(self):
        client = self._mock_client()
        client._session.get_tickers.return_value = {
            'retCode': 0,
            'retMsg': 'OK',
            'result': {
                'list': [{
                    'symbol': 'BTCUSDT',
                    'lastPrice': '67500.5',
                    'bid1Price': '67500.0',
                    'ask1Price': '67501.0',
                    'bid1Size': '1.5',
                    'ask1Size': '2.0',
                    'volume24h': '12345.67',
                }],
            },
        }

        ticker = await client.get_ticker('BTC-USDT')

        assert ticker.inst_id == 'BTC-USDT'
        assert ticker.last == Decimal('67500.5')
        assert ticker.bid == Decimal('67500.0')
        assert ticker.ask == Decimal('67501.0')
        client._session.get_tickers.assert_called_once_with(
            category='spot', symbol='BTCUSDT',
        )

    async def test_get_candlesticks(self):
        client = self._mock_client()
        client._session.get_kline.return_value = {
            'retCode': 0,
            'result': {
                'list': [
                    ['1712000000000', '67000', '67800', '66900', '67500', '1234', '0'],
                    ['1711996400000', '66800', '67100', '66700', '67000', '987', '0'],
                ],
            },
        }

        candles = await client.get_candlesticks('BTC-USDT', bar='15m', limit=2)

        assert len(candles) == 2
        # Sorted ascending
        assert candles[0].timestamp < candles[1].timestamp
        client._session.get_kline.assert_called_once_with(
            category='spot', symbol='BTCUSDT', interval='15', limit=2,
        )

    async def test_get_balance(self):
        client = self._mock_client()
        client._session.get_wallet_balance.return_value = {
            'retCode': 0,
            'result': {
                'list': [{
                    'coin': [
                        {
                            'coin': 'USDT',
                            'walletBalance': '10000',
                            'availableToWithdraw': '8500',
                            'locked': '1500',
                        },
                        {
                            'coin': 'BTC',
                            'walletBalance': '0.5',
                            'availableToWithdraw': '0.5',
                            'locked': '0',
                        },
                    ],
                }],
            },
        }

        balances = await client.get_balance()

        assert len(balances) == 2
        usdt = next(b for b in balances if b.currency == 'USDT')
        assert usdt.total == Decimal('10000')
        assert usdt.available == Decimal('8500')

    async def test_place_order(self):
        client = self._mock_client()
        client._session.place_order.return_value = {
            'retCode': 0,
            'result': {
                'orderId': 'bybit-order-123',
                'orderLinkId': 'client-001',
            },
        }

        order = OrderRequest(
            inst_id='BTC-USDT',
            side=Side.BUY,
            order_type=OrderType.MARKET,
            size=Decimal('0.01'),
            client_order_id='client-001',
        )
        response = await client.place_order(order)

        assert response.order_id == 'bybit-order-123'
        assert response.client_order_id == 'client-001'
        assert response.inst_id == 'BTC-USDT'
        assert response.side == Side.BUY

        # Check Bybit-specific params
        call_args = client._session.place_order.call_args.kwargs
        assert call_args['symbol'] == 'BTCUSDT'
        assert call_args['side'] == 'Buy'
        assert call_args['orderType'] == 'Market'
        assert call_args['qty'] == '0.01'

    async def test_cancel_order(self):
        client = self._mock_client()
        client._session.cancel_order.return_value = {
            'retCode': 0,
            'result': {'orderId': 'order-123'},
        }

        await client.cancel_order('BTC-USDT', 'order-123')

        client._session.cancel_order.assert_called_once_with(
            category='spot', symbol='BTCUSDT', orderId='order-123',
        )

    async def test_place_stop_loss(self):
        client = self._mock_client()
        client._session.place_order.return_value = {
            'retCode': 0,
            'result': {'orderId': 'sl-order-123'},
        }

        order_id = await client.place_stop_loss(
            inst_id='BTC-USDT',
            side=Side.SELL,
            size=Decimal('0.01'),
            trigger_price=Decimal('65000'),
        )

        assert order_id == 'sl-order-123'
        call_args = client._session.place_order.call_args.kwargs
        assert call_args['triggerPrice'] == '65000'
        assert call_args['side'] == 'Sell'

    async def test_place_take_profit(self):
        client = self._mock_client()
        client._session.place_order.return_value = {
            'retCode': 0,
            'result': {'orderId': 'tp-order-123'},
        }

        order_id = await client.place_take_profit(
            inst_id='BTC-USDT',
            side=Side.SELL,
            size=Decimal('0.01'),
            trigger_price=Decimal('70000'),
        )

        assert order_id == 'tp-order-123'

    async def test_validate_connection_success(self):
        client = self._mock_client()
        client._session.get_wallet_balance.return_value = {
            'retCode': 0,
            'result': {
                'list': [{
                    'coin': [{'coin': 'USDT', 'walletBalance': '1000',
                              'availableToWithdraw': '1000', 'locked': '0'}],
                }],
            },
        }

        result = await client.validate_connection()
        assert result is True

    async def test_validate_connection_auth_failure(self):
        client = self._mock_client()
        client._session.get_wallet_balance.return_value = {
            'retCode': 10003,
            'retMsg': 'Invalid API key',
        }

        result = await client.validate_connection()
        assert result is False
