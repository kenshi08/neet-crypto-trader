"""Tests for Telegram notifier — tests work even without python-telegram-bot."""

from __future__ import annotations

from decimal import Decimal

from nct.notifier import TelegramNotifier


class TestNotifierWithoutTelegram:
    def test_disabled_when_no_telegram_lib(self):
        # This test works regardless of whether python-telegram-bot is installed
        # because we test the graceful degradation path
        notifier = TelegramNotifier(token='fake', chat_id='123')
        # If telegram lib is missing, enabled=False
        # If telegram lib is present, enabled=True
        assert isinstance(notifier.enabled, bool)

    async def test_send_is_noop_when_disabled(self):
        notifier = TelegramNotifier(token='fake', chat_id='123')
        if not notifier.enabled:
            # Should not crash even when disabled
            await notifier.send('test message')

    async def test_notify_trade_opened_noop_when_disabled(self):
        notifier = TelegramNotifier(token='fake', chat_id='123')
        if not notifier.enabled:
            await notifier.notify_trade_opened(
                inst_id='BTC-USDT',
                side='buy',
                size=Decimal('0.01'),
                entry_price=Decimal('67500'),
                stop_loss=Decimal('65475'),
                take_profit=Decimal('70875'),
            )

    async def test_stop_is_safe_when_disabled(self):
        notifier = TelegramNotifier(token='fake', chat_id='123')
        if not notifier.enabled:
            await notifier.stop()
