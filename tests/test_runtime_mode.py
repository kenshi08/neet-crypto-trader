"""Tests for runtime mode detection (#39)."""

from __future__ import annotations

from nct.runtime_mode import RunningMode, describe, detect_mode


class TestDetectMode:
    def test_no_api_key_is_paper(self):
        mode = detect_mode(has_api_key=False, demo_mode=True)
        assert mode == RunningMode.PAPER_DRY_RUN

    def test_no_api_key_ignores_demo_flag(self):
        # Even if demo_mode=False, absence of API key means paper
        mode = detect_mode(has_api_key=False, demo_mode=False)
        assert mode == RunningMode.PAPER_DRY_RUN

    def test_api_key_with_demo_is_demo_real_balance(self):
        mode = detect_mode(has_api_key=True, demo_mode=True)
        assert mode == RunningMode.DEMO_REAL_BALANCE

    def test_api_key_without_demo_is_live(self):
        mode = detect_mode(has_api_key=True, demo_mode=False)
        assert mode == RunningMode.LIVE_REAL_MONEY


class TestDescribe:
    def test_paper_description(self):
        desc = describe(RunningMode.PAPER_DRY_RUN)
        assert desc.mode == RunningMode.PAPER_DRY_RUN
        assert desc.is_live is False
        assert 'no real money' in desc.telegram_message.lower()
        assert 'no api key' in desc.log_warning.lower()

    def test_demo_description(self):
        desc = describe(RunningMode.DEMO_REAL_BALANCE)
        assert desc.mode == RunningMode.DEMO_REAL_BALANCE
        assert desc.is_live is False
        assert 'simulated' in desc.telegram_message.lower()

    def test_live_description_marked_live(self):
        desc = describe(RunningMode.LIVE_REAL_MONEY)
        assert desc.mode == RunningMode.LIVE_REAL_MONEY
        assert desc.is_live is True
        # Live mode warnings must be visually distinct
        assert 'live' in desc.telegram_message.lower()
        assert 'real money' in desc.telegram_message.lower()

    def test_only_live_mode_is_flagged_live(self):
        # Safety invariant: is_live must be True ONLY for LIVE_REAL_MONEY
        assert describe(RunningMode.LIVE_REAL_MONEY).is_live is True
        assert describe(RunningMode.DEMO_REAL_BALANCE).is_live is False
        assert describe(RunningMode.PAPER_DRY_RUN).is_live is False


class TestRunningModeEnum:
    def test_values_are_stable_strings(self):
        # These values appear in logs and external monitoring — don't rename
        # them casually. If you need to rename, update the log dashboards too.
        assert RunningMode.PAPER_DRY_RUN.value == 'PAPER_DRY_RUN'
        assert RunningMode.DEMO_REAL_BALANCE.value == 'DEMO_REAL_BALANCE'
        assert RunningMode.LIVE_REAL_MONEY.value == 'LIVE_REAL_MONEY'

    def test_enum_covers_all_cases(self):
        # If a new mode is added, describe() must handle it
        for mode in RunningMode:
            desc = describe(mode)
            assert desc.mode == mode
