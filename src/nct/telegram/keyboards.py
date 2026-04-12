"""Inline keyboard builders and callback data parsing for Telegram UI.

Callback data format: "action:arg1:arg2" (max 64 bytes per Telegram limit).
"""

from __future__ import annotations

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


def parse_callback(data: str) -> tuple[str, list[str]]:
    """Parse 'action:arg1:arg2' into (action, [arg1, arg2])."""
    parts = data.split(':')
    return parts[0], parts[1:]


# ---------------------------------------------------------------------------
# Home / Navigation
# ---------------------------------------------------------------------------

def home_keyboard() -> InlineKeyboardMarkup:
    """Main control panel keyboard."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton('Status', callback_data='nav:status'),
            InlineKeyboardButton('Positions', callback_data='nav:positions'),
            InlineKeyboardButton('Balance', callback_data='nav:balance'),
        ],
        [
            InlineKeyboardButton('Risk', callback_data='nav:risk'),
            InlineKeyboardButton('Pause', callback_data='act:pause'),
            InlineKeyboardButton('Resume', callback_data='act:resume'),
        ],
        [
            InlineKeyboardButton('Emergency', callback_data='nav:emergency'),
        ],
    ])


def back_keyboard(target: str = 'home') -> InlineKeyboardMarkup:
    """Single back button."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton('\u2190 Back', callback_data=f'nav:{target}')],
    ])


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def positions_keyboard(open_pairs: list[str]) -> InlineKeyboardMarkup:
    """One button per open position + back."""
    rows = [
        [InlineKeyboardButton(pair, callback_data=f'pos:{pair}')]
        for pair in open_pairs
    ]
    if not rows:
        rows = [[InlineKeyboardButton('No open positions', callback_data='nav:home')]]
    rows.append([InlineKeyboardButton('\u2190 Back', callback_data='nav:home')])
    return InlineKeyboardMarkup(rows)


def position_detail_keyboard(inst_id: str) -> InlineKeyboardMarkup:
    """Actions for a single position."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton('Explain', callback_data=f'why:{inst_id}'),
            InlineKeyboardButton('Signal', callback_data=f'signal:{inst_id}'),
        ],
        [
            InlineKeyboardButton('Close Position', callback_data=f'close:{inst_id}'),
            InlineKeyboardButton('Refresh', callback_data=f'pos:{inst_id}'),
        ],
        [InlineKeyboardButton('\u2190 Back', callback_data='nav:positions')],
    ])


# ---------------------------------------------------------------------------
# Confirmations
# ---------------------------------------------------------------------------

def confirm_close_keyboard(inst_id: str) -> InlineKeyboardMarkup:
    """Confirm closing a single position."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                'Yes, Close', callback_data=f'confirm:close:{inst_id}',
            ),
            InlineKeyboardButton('Cancel', callback_data=f'pos:{inst_id}'),
        ],
    ])


def confirm_stop_keyboard() -> InlineKeyboardMarkup:
    """Confirm emergency kill switch."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                'Yes, Close All', callback_data='confirm:stop',
            ),
            InlineKeyboardButton('Cancel', callback_data='nav:home'),
        ],
    ])


# ---------------------------------------------------------------------------
# Emergency
# ---------------------------------------------------------------------------

def emergency_keyboard() -> InlineKeyboardMarkup:
    """Emergency actions panel."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton('Close All Positions', callback_data='emergency:stop')],
        [InlineKeyboardButton('Pause Bot', callback_data='act:pause')],
        [InlineKeyboardButton('\u2190 Back', callback_data='nav:home')],
    ])


# ---------------------------------------------------------------------------
# Alert action buttons (attached to notifications)
# ---------------------------------------------------------------------------

def alert_trade_keyboard(inst_id: str) -> InlineKeyboardMarkup:
    """Action buttons attached to trade opened alerts."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton('Explain', callback_data=f'why:{inst_id}'),
            InlineKeyboardButton('Close', callback_data=f'close:{inst_id}'),
        ],
    ])
