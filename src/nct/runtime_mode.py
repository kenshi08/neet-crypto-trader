"""Running mode detection — distinguishes paper / demo / live at a glance.

The bot has three distinct runtime modes which differ in their risk exposure:

- **PAPER_DRY_RUN**: no API key configured. Nothing real, orders simulated
  against an empty balance. Used for local development and CI.
- **DEMO_REAL_BALANCE**: API key with ``demo_mode=True``. Real balance is
  visible (from exchange testnet / demo account), orders are simulated
  locally or sent to an exchange sandbox.
- **LIVE_REAL_MONEY**: API key with ``demo_mode=False``. Real money moves.

The mode must be unmistakable in logs and Telegram so operators can never
confuse a live session for paper trading (or vice versa). See #39.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RunningMode(StrEnum):
    PAPER_DRY_RUN = 'PAPER_DRY_RUN'
    DEMO_REAL_BALANCE = 'DEMO_REAL_BALANCE'
    LIVE_REAL_MONEY = 'LIVE_REAL_MONEY'


@dataclass(frozen=True)
class ModeDescription:
    mode: RunningMode
    log_warning: str
    telegram_message: str
    is_live: bool


_DESCRIPTIONS: dict[RunningMode, ModeDescription] = {
    RunningMode.PAPER_DRY_RUN: ModeDescription(
        mode=RunningMode.PAPER_DRY_RUN,
        log_warning='No API key provided — simulated everything',
        telegram_message='Bot started (paper mode, no real money).',
        is_live=False,
    ),
    RunningMode.DEMO_REAL_BALANCE: ModeDescription(
        mode=RunningMode.DEMO_REAL_BALANCE,
        log_warning='Real balance visible, orders simulated',
        telegram_message='Bot started (demo mode, orders simulated).',
        is_live=False,
    ),
    RunningMode.LIVE_REAL_MONEY: ModeDescription(
        mode=RunningMode.LIVE_REAL_MONEY,
        log_warning='REAL MONEY WILL MOVE — live trading is active',
        telegram_message='*Bot started (LIVE MODE — real money will move).*',
        is_live=True,
    ),
}


def detect_mode(*, has_api_key: bool, demo_mode: bool) -> RunningMode:
    """Determine the running mode from credential state.

    Args:
        has_api_key: True if an API key is configured for the active exchange.
        demo_mode: The active exchange's ``demo_mode`` flag.
    """
    if not has_api_key:
        return RunningMode.PAPER_DRY_RUN
    if demo_mode:
        return RunningMode.DEMO_REAL_BALANCE
    return RunningMode.LIVE_REAL_MONEY


def describe(mode: RunningMode) -> ModeDescription:
    return _DESCRIPTIONS[mode]
