"""Tests for protection plugins — StoplossGuard, MaxDrawdown, CooldownPeriod."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from nct.risk.protections import (
    CooldownPeriod,
    MaxDrawdown,
    ProtectionManager,
    StoplossGuard,
)


class TestStoplossGuard:
    def test_not_locked_initially(self):
        guard = StoplossGuard(trade_limit=3, lookback_seconds=3600, stop_duration_seconds=3600)
        result = guard.check(pair='BTC-USDT')
        assert result.locked is False

    def test_locks_after_reaching_trade_limit(self):
        guard = StoplossGuard(trade_limit=3, lookback_seconds=3600, stop_duration_seconds=3600)
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        for i in range(3):
            guard.record_trade_close(
                pair='BTC-USDT',
                pnl=Decimal('-10'),
                was_stop_loss=True,
                closed_at=now + timedelta(minutes=i),
            )

        result = guard.check(pair='BTC-USDT', now=now + timedelta(minutes=5))
        assert result.locked is True
        assert 'StoplossGuard' in result.reason

    def test_unlocks_after_duration(self):
        guard = StoplossGuard(trade_limit=2, lookback_seconds=3600, stop_duration_seconds=300)
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        for i in range(2):
            guard.record_trade_close(
                pair='BTC-USDT', pnl=Decimal('-10'), was_stop_loss=True,
                closed_at=now + timedelta(minutes=i),
            )

        # Locked immediately after
        assert guard.check(pair='BTC-USDT', now=now + timedelta(minutes=3)).locked is True

        # Unlocked after 5 minutes
        assert guard.check(pair='BTC-USDT', now=now + timedelta(minutes=10)).locked is False

    def test_non_stop_loss_trades_dont_count(self):
        guard = StoplossGuard(trade_limit=2, lookback_seconds=3600, stop_duration_seconds=3600)
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        guard.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('-10'), was_stop_loss=True, closed_at=now,
        )
        guard.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('20'), was_stop_loss=False, closed_at=now,
        )

        assert guard.check(pair='BTC-USDT', now=now).locked is False

    def test_per_pair_mode(self):
        guard = StoplossGuard(
            trade_limit=2, lookback_seconds=3600, stop_duration_seconds=3600, per_pair=True,
        )
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        for i in range(2):
            guard.record_trade_close(
                pair='BTC-USDT', pnl=Decimal('-10'), was_stop_loss=True,
                closed_at=now + timedelta(minutes=i),
            )

        # BTC is locked
        assert guard.check(pair='BTC-USDT', now=now + timedelta(minutes=5)).locked is True
        # ETH is not
        assert guard.check(pair='ETH-USDT', now=now + timedelta(minutes=5)).locked is False

    def test_old_entries_cleaned_up(self):
        guard = StoplossGuard(trade_limit=2, lookback_seconds=60, stop_duration_seconds=60)
        old_time = datetime(2026, 4, 9, 11, 0, tzinfo=UTC)
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        # Old stop losses (outside lookback)
        guard.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('-10'), was_stop_loss=True, closed_at=old_time,
        )
        guard.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('-10'), was_stop_loss=True, closed_at=old_time,
        )

        # Recent one triggers cleanup, old ones are purged
        guard.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('-10'), was_stop_loss=True, closed_at=now,
        )

        assert guard.check(pair='BTC-USDT', now=now).locked is False  # only 1 recent

    def test_reset(self):
        guard = StoplossGuard(trade_limit=1, lookback_seconds=3600, stop_duration_seconds=3600)
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        guard.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('-10'), was_stop_loss=True, closed_at=now,
        )
        assert guard.check(pair='BTC-USDT', now=now).locked is True

        guard.reset()
        assert guard.check(pair='BTC-USDT', now=now).locked is False


class TestMaxDrawdown:
    def test_not_locked_initially(self):
        dd = MaxDrawdown(max_drawdown_usdt=Decimal('50'))
        assert dd.check(pair='BTC-USDT').locked is False

    def test_locks_when_drawdown_exceeded(self):
        dd = MaxDrawdown(max_drawdown_usdt=Decimal('50'))
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        # Win 30, then lose 80 → HWM=30, PnL=-50, drawdown=80
        dd.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('30'), was_stop_loss=False, closed_at=now,
        )
        dd.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('-80'), was_stop_loss=True, closed_at=now,
        )

        assert dd.check(pair='BTC-USDT').locked is True
        assert 'MaxDrawdown' in dd.check(pair='BTC-USDT').reason

    def test_hwm_tracks_correctly(self):
        dd = MaxDrawdown(max_drawdown_usdt=Decimal('100'))
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        dd.record_trade_close(pair='X', pnl=Decimal('50'), was_stop_loss=False, closed_at=now)
        dd.record_trade_close(pair='X', pnl=Decimal('20'), was_stop_loss=False, closed_at=now)
        # HWM = 70
        assert dd.current_drawdown == Decimal(0)

        dd.record_trade_close(pair='X', pnl=Decimal('-30'), was_stop_loss=True, closed_at=now)
        # PnL = 40, HWM = 70, DD = 30
        assert dd.current_drawdown == Decimal('30')
        assert dd.check(pair='X').locked is False

    def test_pure_losses_lock(self):
        dd = MaxDrawdown(max_drawdown_usdt=Decimal('20'))
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        dd.record_trade_close(pair='X', pnl=Decimal('-20'), was_stop_loss=True, closed_at=now)
        # HWM=0, PnL=-20, DD=20
        assert dd.check(pair='X').locked is True

    def test_reset(self):
        dd = MaxDrawdown(max_drawdown_usdt=Decimal('10'))
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        dd.record_trade_close(pair='X', pnl=Decimal('-10'), was_stop_loss=True, closed_at=now)
        assert dd.check(pair='X').locked is True

        dd.reset()
        assert dd.check(pair='X').locked is False


class TestCooldownPeriod:
    def test_not_locked_initially(self):
        cd = CooldownPeriod(cooldown_seconds=300)
        assert cd.check(pair='BTC-USDT').locked is False

    def test_locked_after_trade_close(self):
        cd = CooldownPeriod(cooldown_seconds=300)
        close_time = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        cd.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('10'), was_stop_loss=False, closed_at=close_time,
        )

        # Locked 1 minute later
        assert cd.check(pair='BTC-USDT', now=close_time + timedelta(minutes=1)).locked is True
        # Unlocked 6 minutes later
        assert cd.check(pair='BTC-USDT', now=close_time + timedelta(minutes=6)).locked is False

    def test_per_pair_independent(self):
        cd = CooldownPeriod(cooldown_seconds=300)
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        cd.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('10'), was_stop_loss=False, closed_at=now,
        )

        assert cd.check(pair='BTC-USDT', now=now + timedelta(minutes=1)).locked is True
        assert cd.check(pair='ETH-USDT', now=now + timedelta(minutes=1)).locked is False

    def test_reset(self):
        cd = CooldownPeriod(cooldown_seconds=300)
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        cd.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('10'), was_stop_loss=False, closed_at=now,
        )
        assert cd.check(pair='BTC-USDT', now=now).locked is True

        cd.reset()
        assert cd.check(pair='BTC-USDT', now=now).locked is False


class TestProtectionManager:
    def test_passes_when_all_protections_pass(self):
        pm = ProtectionManager([
            StoplossGuard(trade_limit=5, lookback_seconds=3600, stop_duration_seconds=3600),
            CooldownPeriod(cooldown_seconds=300),
        ])
        assert pm.check(pair='BTC-USDT').locked is False

    def test_fails_on_first_locked_protection(self):
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
        cd = CooldownPeriod(cooldown_seconds=300)
        cd.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('10'), was_stop_loss=False, closed_at=now,
        )

        pm = ProtectionManager([cd])
        result = pm.check(pair='BTC-USDT', now=now + timedelta(minutes=1))
        assert result.locked is True
        assert 'CooldownPeriod' in result.reason

    def test_record_propagates_to_all(self):
        guard = StoplossGuard(trade_limit=1, lookback_seconds=3600, stop_duration_seconds=3600)
        dd = MaxDrawdown(max_drawdown_usdt=Decimal('100'))
        pm = ProtectionManager([guard, dd])
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

        pm.record_trade_close(
            pair='BTC-USDT', pnl=Decimal('-10'), was_stop_loss=True, closed_at=now,
        )

        # Guard should be locked (trade_limit=1)
        assert guard.check(pair='BTC-USDT', now=now).locked is True
        # DD should track the loss
        assert dd.current_drawdown == Decimal('10')

    def test_reset_all(self):
        guard = StoplossGuard(trade_limit=1, lookback_seconds=3600, stop_duration_seconds=3600)
        now = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
        guard.record_trade_close(pair='X', pnl=Decimal('-1'), was_stop_loss=True, closed_at=now)

        pm = ProtectionManager([guard])
        assert pm.check(pair='X', now=now).locked is True

        pm.reset_all()
        assert pm.check(pair='X', now=now).locked is False
