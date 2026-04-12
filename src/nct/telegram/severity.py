"""Alert severity levels and quiet hours filtering for Telegram notifications."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum
from zoneinfo import ZoneInfo


class AlertSeverity(IntEnum):
    """Notification severity — higher value = more important.

    Using IntEnum so comparisons like ``severity >= min_severity`` work naturally.
    """

    LOW = 1       # cooldown notice, iteration stats
    MEDIUM = 2    # daily summary, period reset
    HIGH = 3      # trade opened/closed, limit hit
    CRITICAL = 4  # kill switch, SL failed, reversal, reconciliation mismatch


# Map string names (used in config) to enum values
_SEVERITY_BY_NAME: dict[str, AlertSeverity] = {s.name.lower(): s for s in AlertSeverity}


def parse_severity(name: str) -> AlertSeverity:
    """Parse a severity name string into an AlertSeverity enum value.

    Raises ValueError for unknown names.
    """
    try:
        return _SEVERITY_BY_NAME[name.lower()]
    except KeyError:
        valid = ', '.join(sorted(_SEVERITY_BY_NAME))
        msg = f"Unknown severity '{name}' — must be one of: {valid}"
        raise ValueError(msg) from None


def is_quiet_hours(
    *,
    start_hour: int,
    end_hour: int,
    tz: str = 'UTC',
    now: datetime | None = None,
) -> bool:
    """Check if current time falls within quiet hours window.

    Handles wrap-around midnight (e.g., start=23, end=7 means 23:00-07:00).
    Returns False if start_hour < 0 (disabled).
    """
    if start_hour < 0:
        return False

    if now is None:
        now = datetime.now(UTC)

    try:
        local_tz = ZoneInfo(tz)
    except (KeyError, Exception):
        local_tz = UTC  # type: ignore[assignment]

    local_now = now.astimezone(local_tz)
    hour = local_now.hour

    if start_hour <= end_hour:
        # Same-day window (e.g., 9-17)
        return start_hour <= hour < end_hour
    # Wrap-around midnight (e.g., 23-7)
    return hour >= start_hour or hour < end_hour


def should_send(
    *,
    severity: AlertSeverity,
    min_severity: AlertSeverity,
    quiet_start: int,
    quiet_end: int,
    quiet_tz: str,
    quiet_min_severity: AlertSeverity,
    now: datetime | None = None,
) -> bool:
    """Determine whether an alert should be sent given severity and quiet hours.

    Returns True if the alert passes all filters.
    """
    # Global minimum severity filter
    if severity < min_severity:
        return False

    # Quiet hours filter — only allow alerts at or above quiet_min_severity
    if is_quiet_hours(start_hour=quiet_start, end_hour=quiet_end, tz=quiet_tz, now=now):
        return severity >= quiet_min_severity

    return True
