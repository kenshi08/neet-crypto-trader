"""Tests for alert severity levels and quiet hours filtering."""

from datetime import UTC, datetime

import pytest

from nct.telegram.severity import (
    AlertSeverity,
    is_quiet_hours,
    parse_severity,
    should_send,
)


# -- AlertSeverity ordering -------------------------------------------------

class TestAlertSeverityOrdering:
    def test_severity_ordering(self):
        assert AlertSeverity.LOW < AlertSeverity.MEDIUM
        assert AlertSeverity.MEDIUM < AlertSeverity.HIGH
        assert AlertSeverity.HIGH < AlertSeverity.CRITICAL

    def test_severity_comparison_with_int(self):
        assert AlertSeverity.LOW >= 1
        assert AlertSeverity.CRITICAL >= 4


# -- parse_severity ----------------------------------------------------------

class TestParseSeverity:
    def test_parse_all_valid_names(self):
        assert parse_severity('low') == AlertSeverity.LOW
        assert parse_severity('medium') == AlertSeverity.MEDIUM
        assert parse_severity('high') == AlertSeverity.HIGH
        assert parse_severity('critical') == AlertSeverity.CRITICAL

    def test_parse_case_insensitive(self):
        assert parse_severity('LOW') == AlertSeverity.LOW
        assert parse_severity('Critical') == AlertSeverity.CRITICAL

    def test_parse_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown severity 'debug'"):
            parse_severity('debug')


# -- is_quiet_hours ----------------------------------------------------------

class TestIsQuietHours:
    def test_disabled_when_start_negative(self):
        assert is_quiet_hours(start_hour=-1, end_hour=7) is False

    def test_same_day_window_inside(self):
        now = datetime(2026, 4, 12, 12, 0, tzinfo=UTC)
        assert is_quiet_hours(start_hour=9, end_hour=17, now=now) is True

    def test_same_day_window_outside_before(self):
        now = datetime(2026, 4, 12, 8, 0, tzinfo=UTC)
        assert is_quiet_hours(start_hour=9, end_hour=17, now=now) is False

    def test_same_day_window_outside_after(self):
        now = datetime(2026, 4, 12, 18, 0, tzinfo=UTC)
        assert is_quiet_hours(start_hour=9, end_hour=17, now=now) is False

    def test_wrap_midnight_inside_late(self):
        now = datetime(2026, 4, 12, 23, 30, tzinfo=UTC)
        assert is_quiet_hours(start_hour=23, end_hour=7, now=now) is True

    def test_wrap_midnight_inside_early(self):
        now = datetime(2026, 4, 13, 3, 0, tzinfo=UTC)
        assert is_quiet_hours(start_hour=23, end_hour=7, now=now) is True

    def test_wrap_midnight_outside(self):
        now = datetime(2026, 4, 12, 12, 0, tzinfo=UTC)
        assert is_quiet_hours(start_hour=23, end_hour=7, now=now) is False

    def test_exact_boundary_start_is_inclusive(self):
        now = datetime(2026, 4, 12, 22, 0, tzinfo=UTC)
        assert is_quiet_hours(start_hour=22, end_hour=6, now=now) is True

    def test_exact_boundary_end_is_exclusive(self):
        now = datetime(2026, 4, 13, 6, 0, tzinfo=UTC)
        assert is_quiet_hours(start_hour=22, end_hour=6, now=now) is False

    def test_timezone_conversion(self):
        # Quiet hours 23:00-07:00 Singapore time (UTC+8)
        # UTC 16:00 = SGT 00:00 -> inside quiet hours
        now = datetime(2026, 4, 12, 16, 0, tzinfo=UTC)
        assert is_quiet_hours(
            start_hour=23, end_hour=7, tz='Asia/Singapore', now=now,
        ) is True

    def test_timezone_conversion_outside(self):
        # Quiet hours 23:00-07:00 Singapore time (UTC+8)
        # UTC 06:00 = SGT 14:00 -> outside quiet hours
        now = datetime(2026, 4, 12, 6, 0, tzinfo=UTC)
        assert is_quiet_hours(
            start_hour=23, end_hour=7, tz='Asia/Singapore', now=now,
        ) is False

    def test_invalid_timezone_falls_back_to_utc(self):
        now = datetime(2026, 4, 12, 3, 0, tzinfo=UTC)
        result = is_quiet_hours(start_hour=1, end_hour=5, tz='Invalid/Zone', now=now)
        assert result is True


# -- should_send --------------------------------------------------------------

class TestShouldSend:
    def _defaults(self, **overrides):
        params = {
            'severity': AlertSeverity.HIGH,
            'min_severity': AlertSeverity.LOW,
            'quiet_start': -1,
            'quiet_end': -1,
            'quiet_tz': 'UTC',
            'quiet_min_severity': AlertSeverity.CRITICAL,
            'now': None,
        }
        params.update(overrides)
        return params

    def test_high_severity_passes_default(self):
        assert should_send(**self._defaults()) is True

    def test_low_below_minimum_is_suppressed(self):
        assert should_send(**self._defaults(
            severity=AlertSeverity.LOW,
            min_severity=AlertSeverity.MEDIUM,
        )) is False

    def test_medium_equals_minimum_passes(self):
        assert should_send(**self._defaults(
            severity=AlertSeverity.MEDIUM,
            min_severity=AlertSeverity.MEDIUM,
        )) is True

    def test_critical_always_passes_during_quiet_hours(self):
        now = datetime(2026, 4, 12, 2, 0, tzinfo=UTC)
        assert should_send(**self._defaults(
            severity=AlertSeverity.CRITICAL,
            quiet_start=23,
            quiet_end=7,
            now=now,
        )) is True

    def test_high_suppressed_during_quiet_hours(self):
        now = datetime(2026, 4, 12, 2, 0, tzinfo=UTC)
        assert should_send(**self._defaults(
            severity=AlertSeverity.HIGH,
            quiet_start=23,
            quiet_end=7,
            now=now,
        )) is False

    def test_high_passes_outside_quiet_hours(self):
        now = datetime(2026, 4, 12, 12, 0, tzinfo=UTC)
        assert should_send(**self._defaults(
            severity=AlertSeverity.HIGH,
            quiet_start=23,
            quiet_end=7,
            now=now,
        )) is True

    def test_quiet_hours_disabled_passes_everything(self):
        now = datetime(2026, 4, 12, 2, 0, tzinfo=UTC)
        assert should_send(**self._defaults(
            severity=AlertSeverity.LOW,
            quiet_start=-1,
            quiet_end=7,
            now=now,
        )) is True

    def test_quiet_min_severity_customizable(self):
        now = datetime(2026, 4, 12, 2, 0, tzinfo=UTC)
        assert should_send(**self._defaults(
            severity=AlertSeverity.HIGH,
            quiet_start=23,
            quiet_end=7,
            quiet_min_severity=AlertSeverity.HIGH,
            now=now,
        )) is True
