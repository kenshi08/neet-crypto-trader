"""Tests for exchange data models and parsers."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from nct.exchange.models import (
    OrderStatus,
    Side,
    parse_balance,
    parse_candle,
    parse_order_response,
    parse_position,
    parse_ticker,
)


class TestParseTicker:
    def test_parses_all_fields(self, sample_ticker_data):
        ticker = parse_ticker(sample_ticker_data)

        assert ticker.inst_id == 'BTC-USDT'
        assert ticker.last == Decimal('67500.5')
        assert ticker.bid == Decimal('67500.0')
        assert ticker.ask == Decimal('67501.0')
        assert ticker.bid_size == Decimal('1.5')
        assert ticker.ask_size == Decimal('2.0')
        assert ticker.volume_24h == Decimal('12345.67')
        assert isinstance(ticker.timestamp, datetime)
        assert ticker.timestamp.tzinfo == UTC

    def test_mid_price(self, sample_ticker_data):
        ticker = parse_ticker(sample_ticker_data)
        assert ticker.mid == Decimal('67500.5')

    def test_spread(self, sample_ticker_data):
        ticker = parse_ticker(sample_ticker_data)
        spread = ticker.spread
        assert spread > 0
        assert spread < Decimal('0.001')  # very tight spread

    def test_ticker_is_immutable(self, sample_ticker_data):
        ticker = parse_ticker(sample_ticker_data)
        with __import__('pytest').raises(AttributeError):
            ticker.last = Decimal('99999')  # type: ignore[misc]


class TestParseCandle:
    def test_parses_raw_array(self, sample_candle_raw):
        candle = parse_candle('BTC-USDT', sample_candle_raw)

        assert candle.open == Decimal('67000.0')
        assert candle.high == Decimal('67800.0')
        assert candle.low == Decimal('66900.0')
        assert candle.close == Decimal('67500.5')
        assert candle.volume == Decimal('1234.56')
        assert isinstance(candle.timestamp, datetime)

    def test_bullish_candle(self, sample_candle_raw):
        candle = parse_candle('BTC-USDT', sample_candle_raw)
        assert candle.is_bullish is True  # close > open

    def test_bearish_candle(self):
        raw = ['1712000000000', '67500.0', '67800.0', '66900.0', '67000.0', '1234.56']
        candle = parse_candle('BTC-USDT', raw)
        assert candle.is_bullish is False

    def test_body_size(self, sample_candle_raw):
        candle = parse_candle('BTC-USDT', sample_candle_raw)
        assert candle.body_size == Decimal('500.5')


class TestParseOrderResponse:
    def test_parses_filled_order(self, sample_order_response_data):
        response = parse_order_response(sample_order_response_data)

        assert response.order_id == '123456789'
        assert response.client_order_id == 'client-001'
        assert response.status == OrderStatus.FILLED
        assert response.inst_id == 'BTC-USDT'
        assert response.side == Side.BUY
        assert response.size == Decimal('0.01')
        assert response.filled_size == Decimal('0.01')
        assert response.avg_fill_price == Decimal('67500.0')
        assert response.fee == Decimal('-0.675')

    def test_parses_pending_order(self):
        data = {
            'ordId': '999',
            'clOrdId': '',
            'state': 'live',
            'instId': 'ETH-USDT',
            'side': 'sell',
            'sz': '1.0',
            'px': '3500.0',
        }
        response = parse_order_response(data)
        assert response.status == OrderStatus.PENDING
        assert response.side == Side.SELL

    def test_parses_cancelled_order(self):
        data = {
            'ordId': '888',
            'clOrdId': '',
            'state': 'canceled',
            'instId': 'SOL-USDT',
            'side': 'buy',
            'sz': '10',
        }
        response = parse_order_response(data)
        assert response.status == OrderStatus.CANCELLED

    def test_dry_run_flag(self, sample_order_response_data):
        response = parse_order_response(sample_order_response_data, is_dry_run=True)
        assert response.is_dry_run is True

    def test_order_response_is_immutable(self, sample_order_response_data):
        response = parse_order_response(sample_order_response_data)
        with __import__('pytest').raises(AttributeError):
            response.order_id = 'hacked'  # type: ignore[misc]


class TestParsePosition:
    def test_parses_position(self, sample_position_data):
        pos = parse_position(sample_position_data)

        assert pos.inst_id == 'BTC-USDT'
        assert pos.side == 'net'
        assert pos.size == Decimal('0.01')
        assert pos.avg_price == Decimal('67500.0')
        assert pos.unrealized_pnl == Decimal('50.5')
        assert pos.leverage == Decimal('1')

    def test_defaults_to_net_side(self):
        data = {
            'instId': 'ETH-USDT',
            'posSide': 'invalid',
            'pos': '5',
            'avgPx': '3500',
        }
        pos = parse_position(data)
        assert pos.side == 'net'


class TestParseBalance:
    def test_parses_balance(self, sample_balance_data):
        bal = parse_balance(sample_balance_data)

        assert bal.currency == 'USDT'
        assert bal.total == Decimal('10000.0')
        assert bal.available == Decimal('8500.0')
        assert bal.frozen == Decimal('1500.0')

    def test_balance_is_immutable(self, sample_balance_data):
        bal = parse_balance(sample_balance_data)
        with __import__('pytest').raises(AttributeError):
            bal.total = Decimal('0')  # type: ignore[misc]
