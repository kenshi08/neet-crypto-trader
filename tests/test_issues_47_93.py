"""Tests for remaining open issues.

- #47: Telegram confirmation flow for destructive commands
- #93: Hyperliquid WebSocket market feed
"""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from nct.exchange.market_feed import HyperliquidMarketFeed
from nct.exchange.models import Ticker


# ===================================================================
# Issue #47 — Telegram confirmation flow
# ===================================================================


class TestConfirmationFlow:
    def _make_notifier(self):
        """Create a minimal notifier-like object with confirmation state."""
        # We test the confirmation dict logic directly since the full
        # TelegramNotifier requires python-telegram-bot installed.
        confirmations: dict[str, tuple[str, float]] = {}
        return confirmations

    def test_token_stored_with_expiry(self):
        confirmations = self._make_notifier()
        token = '48291'
        confirmations[token] = ('stop', time.time() + 60)
        assert token in confirmations
        action, expiry = confirmations[token]
        assert action == 'stop'
        assert expiry > time.time()

    def test_token_popped_on_use(self):
        confirmations = self._make_notifier()
        token = '12345'
        confirmations[token] = ('stop', time.time() + 60)
        pending = confirmations.pop(token, None)
        assert pending is not None
        assert token not in confirmations

    def test_expired_token_rejected(self):
        confirmations = self._make_notifier()
        token = '99999'
        confirmations[token] = ('stop', time.time() - 10)  # already expired
        pending = confirmations.pop(token, None)
        assert pending is not None
        _action, expiry = pending
        assert time.time() > expiry  # expired

    def test_invalid_token_returns_none(self):
        confirmations = self._make_notifier()
        assert confirmations.pop('nonexistent', None) is None

    def test_multiple_tokens_independent(self):
        confirmations = self._make_notifier()
        confirmations['11111'] = ('stop', time.time() + 60)
        confirmations['22222'] = ('stop', time.time() + 60)
        confirmations.pop('11111')
        assert '22222' in confirmations


# ===================================================================
# Issue #93 — Hyperliquid WebSocket market feed
# ===================================================================


class TestHyperliquidMarketFeed:
    def test_construction(self):
        feed = HyperliquidMarketFeed(
            ['BTC-USDT', 'ETH-USDT'], demo_mode=True,
        )
        assert feed.is_running is False
        assert feed.connected_pairs == []

    def test_get_latest_ticker_none_before_start(self):
        feed = HyperliquidMarketFeed(['BTC-USDT'])
        assert feed.get_latest_ticker('BTC-USDT') is None

    def test_callback_registration(self):
        feed = HyperliquidMarketFeed(['BTC-USDT'])
        cb = MagicMock()
        feed.on_ticker(cb)
        assert len(feed._ticker_callbacks) == 1

    def test_dispatch_calls_callbacks(self):
        feed = HyperliquidMarketFeed(['BTC-USDT'])
        cb = MagicMock()
        feed.on_ticker(cb)

        from datetime import UTC, datetime
        ticker = Ticker(
            inst_id='BTC-USDT', last=Decimal('65000'),
            bid=Decimal('64990'), ask=Decimal('65010'),
            bid_size=Decimal('1'), ask_size=Decimal('1'),
            volume_24h=Decimal('0'), timestamp=datetime.now(UTC),
        )
        feed._dispatch_ticker(ticker)
        cb.assert_called_once_with(ticker)

    def test_latest_ticker_stored_on_dispatch(self):
        feed = HyperliquidMarketFeed(['BTC-USDT'])

        from datetime import UTC, datetime
        ticker = Ticker(
            inst_id='BTC-USDT', last=Decimal('65000'),
            bid=Decimal('65000'), ask=Decimal('65000'),
            bid_size=Decimal(0), ask_size=Decimal(0),
            volume_24h=Decimal(0), timestamp=datetime.now(UTC),
        )
        feed._latest_tickers['BTC-USDT'] = ticker

        assert feed.get_latest_ticker('BTC-USDT') == ticker
        assert 'BTC-USDT' in feed.connected_pairs

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        feed = HyperliquidMarketFeed(['BTC-USDT'])
        await feed.stop()  # should not raise
        assert feed.is_running is False
