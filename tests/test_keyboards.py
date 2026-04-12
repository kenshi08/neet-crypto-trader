"""Tests for Telegram inline keyboard builders and callback parsing."""

import pytest

from nct.telegram.keyboards import (
    alert_trade_keyboard,
    back_keyboard,
    confirm_close_keyboard,
    confirm_stop_keyboard,
    emergency_keyboard,
    home_keyboard,
    parse_callback,
    position_detail_keyboard,
    positions_keyboard,
)


class TestParseCallback:
    def test_simple_action(self):
        action, args = parse_callback('nav:home')
        assert action == 'nav'
        assert args == ['home']

    def test_action_with_pair(self):
        action, args = parse_callback('close:BTC-USD')
        assert action == 'close'
        assert args == ['BTC-USD']

    def test_action_with_multiple_args(self):
        action, args = parse_callback('confirm:close:ETH-USD')
        assert action == 'confirm'
        assert args == ['close', 'ETH-USD']

    def test_action_only(self):
        action, args = parse_callback('cancel')
        assert action == 'cancel'
        assert args == []

    def test_callback_data_within_64_bytes(self):
        # Worst-case: confirm:close:VERY-LONG-PAIR-NAME
        data = 'confirm:close:1000PEPE-USDT'
        assert len(data.encode('utf-8')) <= 64


class TestHomeKeyboard:
    def test_home_has_buttons(self):
        kb = home_keyboard()
        assert len(kb.inline_keyboard) == 3  # 3 rows
        all_buttons = [btn for row in kb.inline_keyboard for btn in row]
        assert len(all_buttons) == 7

    def test_home_button_callbacks(self):
        kb = home_keyboard()
        callbacks = {btn.callback_data for row in kb.inline_keyboard for btn in row}
        assert 'nav:status' in callbacks
        assert 'nav:positions' in callbacks
        assert 'nav:emergency' in callbacks


class TestPositionsKeyboard:
    def test_with_open_pairs(self):
        kb = positions_keyboard(['BTC-USD', 'ETH-USD'])
        # 2 pair buttons + 1 back button = 3 rows
        assert len(kb.inline_keyboard) == 3
        assert kb.inline_keyboard[0][0].callback_data == 'pos:BTC-USD'
        assert kb.inline_keyboard[1][0].callback_data == 'pos:ETH-USD'

    def test_empty_positions(self):
        kb = positions_keyboard([])
        assert len(kb.inline_keyboard) == 2  # "No positions" + back
        assert 'No open positions' in kb.inline_keyboard[0][0].text


class TestPositionDetailKeyboard:
    def test_has_explain_and_close(self):
        kb = position_detail_keyboard('BTC-USD')
        callbacks = {btn.callback_data for row in kb.inline_keyboard for btn in row}
        assert 'why:BTC-USD' in callbacks
        assert 'close:BTC-USD' in callbacks
        assert 'signal:BTC-USD' in callbacks
        assert 'nav:positions' in callbacks  # back button


class TestConfirmKeyboards:
    def test_close_confirm(self):
        kb = confirm_close_keyboard('SOL-USD')
        callbacks = {btn.callback_data for row in kb.inline_keyboard for btn in row}
        assert 'confirm:close:SOL-USD' in callbacks

    def test_stop_confirm(self):
        kb = confirm_stop_keyboard()
        callbacks = {btn.callback_data for row in kb.inline_keyboard for btn in row}
        assert 'confirm:stop' in callbacks


class TestEmergencyKeyboard:
    def test_has_close_all_and_pause(self):
        kb = emergency_keyboard()
        callbacks = {btn.callback_data for row in kb.inline_keyboard for btn in row}
        assert 'emergency:stop' in callbacks
        assert 'act:pause' in callbacks


class TestAlertKeyboard:
    def test_trade_alert_buttons(self):
        kb = alert_trade_keyboard('BTC-USD')
        callbacks = {btn.callback_data for row in kb.inline_keyboard for btn in row}
        assert 'why:BTC-USD' in callbacks
        assert 'close:BTC-USD' in callbacks


class TestBackKeyboard:
    def test_default_back_to_home(self):
        kb = back_keyboard()
        assert kb.inline_keyboard[0][0].callback_data == 'nav:home'

    def test_custom_back_target(self):
        kb = back_keyboard('positions')
        assert kb.inline_keyboard[0][0].callback_data == 'nav:positions'
