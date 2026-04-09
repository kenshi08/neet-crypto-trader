"""Tests for the WebSocket market feed."""

from __future__ import annotations

import json

from nct.exchange.market_feed import MarketFeed


class TestMarketFeedInit:
    def test_demo_mode_url(self):
        feed = MarketFeed(['BTC-USDT'], demo_mode=True)
        assert 'wspap' in feed._url

    def test_live_mode_url(self):
        feed = MarketFeed(['BTC-USDT'], demo_mode=False)
        assert 'ws.okx.com' in feed._url

    def test_registers_callback(self):
        feed = MarketFeed(['BTC-USDT'])
        assert len(feed._ticker_callbacks) == 0

        feed.on_ticker(lambda t: None)
        assert len(feed._ticker_callbacks) == 1

    def test_no_latest_ticker_initially(self):
        feed = MarketFeed(['BTC-USDT'])
        assert feed.get_latest_ticker('BTC-USDT') is None

    def test_not_running_initially(self):
        feed = MarketFeed(['BTC-USDT'])
        assert feed.is_running is False


class TestMarketFeedMessageParsing:
    def test_parses_ticker_message(self):
        feed = MarketFeed(['BTC-USDT'])
        received = []
        feed.on_ticker(lambda t: received.append(t))

        msg = json.dumps({
            'arg': {'channel': 'tickers', 'instId': 'BTC-USDT'},
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
        })

        feed._on_message(msg)

        assert len(received) == 1
        assert received[0].inst_id == 'BTC-USDT'
        assert received[0].last.__str__() == '67500.5'

    def test_stores_latest_ticker(self):
        feed = MarketFeed(['BTC-USDT'])

        msg = json.dumps({
            'arg': {'channel': 'tickers', 'instId': 'BTC-USDT'},
            'data': [{
                'instId': 'BTC-USDT',
                'last': '67500.5',
                'bidPx': '67500.0',
                'askPx': '67501.0',
                'ts': '1712000000000',
            }],
        })

        feed._on_message(msg)

        ticker = feed.get_latest_ticker('BTC-USDT')
        assert ticker is not None
        assert ticker.inst_id == 'BTC-USDT'

    def test_ignores_subscribe_event(self):
        feed = MarketFeed(['BTC-USDT'])
        received = []
        feed.on_ticker(lambda t: received.append(t))

        msg = json.dumps({'event': 'subscribe', 'arg': {'channel': 'tickers'}})
        feed._on_message(msg)
        assert len(received) == 0

    def test_handles_error_event(self):
        feed = MarketFeed(['BTC-USDT'])
        msg = json.dumps({'event': 'error', 'code': '123', 'msg': 'bad'})
        # Should not crash
        feed._on_message(msg)

    def test_handles_invalid_json(self):
        feed = MarketFeed(['BTC-USDT'])
        feed._on_message('not json at all')
        # Should not crash

    def test_handles_missing_data(self):
        feed = MarketFeed(['BTC-USDT'])
        feed._on_message(json.dumps({'arg': {'channel': 'tickers'}}))
        # No 'data' key — should not crash

    def test_multiple_tickers_in_one_message(self):
        feed = MarketFeed(['BTC-USDT', 'ETH-USDT'])

        msg = json.dumps({
            'arg': {'channel': 'tickers'},
            'data': [
                {
                    'instId': 'BTC-USDT', 'last': '67500', 'bidPx': '67499',
                    'askPx': '67501', 'ts': '1712000000000',
                },
                {
                    'instId': 'ETH-USDT', 'last': '3500', 'bidPx': '3499',
                    'askPx': '3501', 'ts': '1712000000000',
                },
            ],
        })

        feed._on_message(msg)

        assert feed.get_latest_ticker('BTC-USDT') is not None
        assert feed.get_latest_ticker('ETH-USDT') is not None


class TestMarketFeedProperties:
    def test_connected_pairs_empty_initially(self):
        feed = MarketFeed(['BTC-USDT'])
        assert feed.connected_pairs == []

    def test_connected_pairs_after_ticker(self):
        feed = MarketFeed(['BTC-USDT'])
        msg = json.dumps({
            'arg': {'channel': 'tickers'},
            'data': [{
                'instId': 'BTC-USDT', 'last': '67500', 'bidPx': '67499',
                'askPx': '67501', 'ts': '1712000000000',
            }],
        })
        feed._on_message(msg)
        assert 'BTC-USDT' in feed.connected_pairs
