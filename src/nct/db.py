"""SQLite database layer — schema, persistence helpers."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import aiosqlite
import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL DEFAULT '',
    inst_id TEXT NOT NULL,
    side TEXT NOT NULL,
    size TEXT NOT NULL,
    entry_price TEXT NOT NULL,
    exit_price TEXT,
    pnl TEXT,
    fee TEXT NOT NULL DEFAULT '0',
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    is_dry_run INTEGER NOT NULL DEFAULT 1,
    strategy TEXT,
    signal_confidence REAL,
    stop_loss_price TEXT,
    take_profit_price TEXT
);

CREATE TABLE IF NOT EXISTS budget_periods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL DEFAULT '',
    period_type TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    capital_deployed TEXT NOT NULL DEFAULT '0',
    realized_pnl TEXT NOT NULL DEFAULT '0',
    trade_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS daily_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL DEFAULT '',
    date TEXT NOT NULL,
    realized_pnl TEXT NOT NULL DEFAULT '0',
    trade_count INTEGER NOT NULL DEFAULT 0,
    stop_loss_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(exchange, date)
);

CREATE TABLE IF NOT EXISTS executor_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER,
    action TEXT NOT NULL,
    details TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);

CREATE TABLE IF NOT EXISTS trade_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL,
    inst_id TEXT NOT NULL,
    strategy TEXT,
    signal_confidence REAL,
    intended_entry_price TEXT,
    actual_entry_price TEXT,
    entry_slippage_pct REAL,
    intended_size TEXT,
    actual_filled_size TEXT,
    fill_rate REAL,
    exit_reason TEXT,
    exit_slippage_pct REAL,
    entry_latency_ms REAL,
    sl_placement_latency_ms REAL,
    tp_placement_latency_ms REAL,
    order_retry_count INTEGER DEFAULT 0,
    fees_paid TEXT,
    opened_at TEXT,
    closed_at TEXT,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);

CREATE TABLE IF NOT EXISTS telegram_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT NOT NULL,
    command TEXT NOT NULL,
    args TEXT,
    result TEXT NOT NULL DEFAULT 'success'
);
"""


# ---------------------------------------------------------------------------
# Database connection management
# ---------------------------------------------------------------------------


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        log.info('db_connected', path=str(self._db_path))

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            log.info('db_closed')

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            msg = 'Database not connected. Call connect() first.'
            raise RuntimeError(msg)
        return self._conn

    # -- Trade operations -----------------------------------------------

    async def insert_trade(
        self,
        *,
        inst_id: str,
        side: str,
        size: Decimal,
        entry_price: Decimal,
        fee: Decimal,
        opened_at: datetime,
        is_dry_run: bool,
        strategy: str = '',
        signal_confidence: float = 0.0,
        stop_loss_price: Decimal | None = None,
        take_profit_price: Decimal | None = None,
        exchange: str = '',
    ) -> int:
        cursor = await self.conn.execute(
            """INSERT INTO trades
               (exchange, inst_id, side, size, entry_price, fee, opened_at,
                is_dry_run, strategy, signal_confidence,
                stop_loss_price, take_profit_price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                exchange, inst_id, side, str(size), str(entry_price), str(fee),
                opened_at.isoformat(), int(is_dry_run),
                strategy, signal_confidence,
                str(stop_loss_price) if stop_loss_price else None,
                str(take_profit_price) if take_profit_price else None,
            ),
        )
        await self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def close_trade(
        self,
        trade_id: int,
        *,
        exit_price: Decimal,
        pnl: Decimal,
        closed_at: datetime,
    ) -> None:
        await self.conn.execute(
            """UPDATE trades SET exit_price = ?, pnl = ?, closed_at = ? WHERE id = ?""",
            (str(exit_price), str(pnl), closed_at.isoformat(), trade_id),
        )
        await self.conn.commit()

    async def get_open_trades(self, exchange: str = '') -> list[dict]:
        if exchange:
            cursor = await self.conn.execute(
                "SELECT * FROM trades WHERE closed_at IS NULL AND exchange = ? "
                "ORDER BY opened_at DESC",
                (exchange,),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM trades WHERE closed_at IS NULL ORDER BY opened_at DESC"
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # -- Budget period operations ---------------------------------------

    async def get_active_budget_period(
        self, period_type: str, exchange: str = '',
    ) -> dict | None:
        cursor = await self.conn.execute(
            "SELECT * FROM budget_periods "
            "WHERE period_type = ? AND exchange = ? AND status = 'active' "
            "ORDER BY start_date DESC LIMIT 1",
            (period_type, exchange),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def create_budget_period(
        self,
        *,
        period_type: str,
        start_date: datetime,
        end_date: datetime,
        exchange: str = '',
    ) -> int:
        cursor = await self.conn.execute(
            """INSERT INTO budget_periods (exchange, period_type, start_date, end_date)
               VALUES (?, ?, ?, ?)""",
            (exchange, period_type, start_date.isoformat(), end_date.isoformat()),
        )
        await self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def update_budget_period(
        self,
        period_id: int,
        *,
        capital_deployed: Decimal,
        realized_pnl: Decimal,
        trade_count: int,
    ) -> None:
        await self.conn.execute(
            """UPDATE budget_periods
               SET capital_deployed = ?, realized_pnl = ?, trade_count = ?
               WHERE id = ?""",
            (str(capital_deployed), str(realized_pnl), trade_count, period_id),
        )
        await self.conn.commit()

    async def close_budget_period(self, period_id: int) -> None:
        await self.conn.execute(
            "UPDATE budget_periods SET status = 'closed' WHERE id = ?",
            (period_id,),
        )
        await self.conn.commit()

    # -- Daily stats operations -----------------------------------------

    async def get_daily_stats(self, date_str: str, exchange: str = '') -> dict | None:
        cursor = await self.conn.execute(
            "SELECT * FROM daily_stats WHERE date = ? AND exchange = ?",
            (date_str, exchange),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def upsert_daily_stats(
        self,
        *,
        date_str: str,
        realized_pnl: Decimal,
        trade_count: int,
        stop_loss_count: int,
        exchange: str = '',
    ) -> None:
        await self.conn.execute(
            """INSERT INTO daily_stats
               (exchange, date, realized_pnl, trade_count, stop_loss_count)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(exchange, date) DO UPDATE SET
                   realized_pnl = ?,
                   trade_count = ?,
                   stop_loss_count = ?""",
            (
                exchange, date_str, str(realized_pnl), trade_count, stop_loss_count,
                str(realized_pnl), trade_count, stop_loss_count,
            ),
        )
        await self.conn.commit()

    # -- Executor log ---------------------------------------------------

    async def log_action(
        self,
        *,
        action: str,
        details: str = '',
        trade_id: int | None = None,
        timestamp: datetime,
    ) -> None:
        await self.conn.execute(
            """INSERT INTO executor_log (trade_id, action, details, timestamp)
               VALUES (?, ?, ?, ?)""",
            (trade_id, action, details, timestamp.isoformat()),
        )
        await self.conn.commit()


    # -- Trade analytics ---------------------------------------------------

    async def record_trade_analytics_open(
        self,
        *,
        trade_id: int,
        inst_id: str,
        strategy: str = '',
        signal_confidence: float = 0.0,
        intended_entry_price: str = '',
        actual_entry_price: str = '',
        entry_slippage_pct: float = 0.0,
        intended_size: str = '',
        actual_filled_size: str = '',
        fill_rate: float = 1.0,
        entry_latency_ms: float = 0.0,
        sl_placement_latency_ms: float = 0.0,
        tp_placement_latency_ms: float = 0.0,
        fees_paid: str = '0',
        opened_at: datetime | None = None,
    ) -> None:
        ts = opened_at.isoformat() if opened_at else ''
        await self.conn.execute(
            """INSERT INTO trade_analytics
               (trade_id, inst_id, strategy, signal_confidence,
                intended_entry_price, actual_entry_price, entry_slippage_pct,
                intended_size, actual_filled_size, fill_rate,
                entry_latency_ms, sl_placement_latency_ms, tp_placement_latency_ms,
                fees_paid, opened_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id, inst_id, strategy, signal_confidence,
                intended_entry_price, actual_entry_price, entry_slippage_pct,
                intended_size, actual_filled_size, fill_rate,
                entry_latency_ms, sl_placement_latency_ms, tp_placement_latency_ms,
                fees_paid, ts,
            ),
        )
        await self.conn.commit()

    async def record_trade_analytics_close(
        self,
        *,
        trade_id: int,
        exit_reason: str = '',
        exit_slippage_pct: float = 0.0,
        closed_at: datetime | None = None,
    ) -> None:
        ts = closed_at.isoformat() if closed_at else ''
        await self.conn.execute(
            """UPDATE trade_analytics
               SET exit_reason = ?, exit_slippage_pct = ?, closed_at = ?
               WHERE trade_id = ?""",
            (exit_reason, exit_slippage_pct, ts, trade_id),
        )
        await self.conn.commit()

    async def get_trade_analytics_summary(
        self, *, days: int | None = None,
    ) -> dict:
        """Aggregate trade analytics for /stats command."""
        where = ''
        params: tuple = ()
        if days is not None:
            where = "WHERE a.opened_at >= datetime('now', ?)"
            params = (f'-{days} days',)

        cursor = await self.conn.execute(
            f"""SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN t.pnl IS NOT NULL AND CAST(t.pnl AS REAL) > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN t.pnl IS NOT NULL AND CAST(t.pnl AS REAL) <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(CAST(t.pnl AS REAL)) as total_pnl,
                AVG(a.entry_slippage_pct) as avg_slippage,
                AVG(a.entry_latency_ms) as avg_entry_latency_ms,
                AVG(a.fill_rate) as avg_fill_rate,
                SUM(CAST(a.fees_paid AS REAL)) as total_fees
            FROM trade_analytics a
            LEFT JOIN trades t ON a.trade_id = t.id
            {where}""",
            params,
        )
        row = await cursor.fetchone()
        if not row:
            return {}
        return dict(row)

    async def get_trade_analytics_by_pair(
        self, *, days: int | None = None,
    ) -> list[dict]:
        """Per-pair analytics breakdown."""
        where = ''
        params: tuple = ()
        if days is not None:
            where = "WHERE a.opened_at >= datetime('now', ?)"
            params = (f'-{days} days',)

        cursor = await self.conn.execute(
            f"""SELECT
                a.inst_id,
                COUNT(*) as trades,
                SUM(CASE WHEN CAST(t.pnl AS REAL) > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CAST(t.pnl AS REAL)) as pnl
            FROM trade_analytics a
            LEFT JOIN trades t ON a.trade_id = t.id
            {where}
            GROUP BY a.inst_id
            ORDER BY pnl DESC""",
            params,
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_trade_analytics_by_exit_reason(
        self, *, days: int | None = None,
    ) -> list[dict]:
        """P&L breakdown by exit reason."""
        where = ''
        params: tuple = ()
        if days is not None:
            where = "WHERE a.opened_at >= datetime('now', ?)"
            params = (f'-{days} days',)

        cursor = await self.conn.execute(
            f"""SELECT
                a.exit_reason,
                COUNT(*) as trades,
                SUM(CAST(t.pnl AS REAL)) as pnl
            FROM trade_analytics a
            LEFT JOIN trades t ON a.trade_id = t.id
            {where}
            GROUP BY a.exit_reason
            ORDER BY trades DESC""",
            params,
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # -- Risk-adjusted metrics ---------------------------------------------

    async def get_risk_adjusted_metrics(
        self, *, days: int | None = None,
    ) -> dict:
        """Compute risk-adjusted performance metrics from closed trades.

        Returns dict with: sharpe_ratio, profit_factor, avg_win, avg_loss,
        expectancy, max_consecutive_losses.
        """
        where = 'WHERE t.closed_at IS NOT NULL'
        params: tuple = ()
        if days is not None:
            where += " AND t.closed_at >= datetime('now', ?)"
            params = (f'-{days} days',)

        cursor = await self.conn.execute(
            f"""SELECT CAST(t.pnl AS REAL) as pnl
            FROM trades t
            {where}
            ORDER BY t.closed_at ASC""",
            params,
        )
        rows = await cursor.fetchall()

        pnls = [r['pnl'] for r in rows if r['pnl'] is not None]
        if not pnls:
            return {
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'expectancy': 0.0,
                'max_consecutive_losses': 0,
                'trade_count': 0,
            }

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float('inf')
        )

        # Sharpe ratio (annualized, assuming ~365 trading days for crypto)
        mean_pnl = sum(pnls) / len(pnls)
        if len(pnls) >= 2:
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std_pnl = variance ** 0.5
            sharpe = (mean_pnl / std_pnl * (365 ** 0.5)) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        # Expectancy = avg_win * win_rate - avg_loss * loss_rate
        win_rate = len(wins) / len(pnls) if pnls else 0
        loss_rate = len(losses) / len(pnls) if pnls else 0
        expectancy = avg_win * win_rate + avg_loss * loss_rate  # avg_loss is negative

        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for p in pnls:
            if p <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        return {
            'sharpe_ratio': round(sharpe, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'expectancy': round(expectancy, 4),
            'max_consecutive_losses': max_consec,
            'trade_count': len(pnls),
        }

    # -- Telegram command log ---------------------------------------------

    async def log_telegram_command(
        self,
        *,
        user_id: str,
        command: str,
        args: str = '',
        result: str = 'success',
        timestamp: datetime | None = None,
    ) -> None:
        ts = timestamp or datetime.now()
        await self.conn.execute(
            """INSERT INTO telegram_commands (timestamp, user_id, command, args, result)
               VALUES (?, ?, ?, ?, ?)""",
            (ts.isoformat(), user_id, command, args, result),
        )
        await self.conn.commit()


def get_db_path(*, is_dry_run: bool) -> Path:
    """Return the database path based on trading mode."""
    name = 'nct_paper.sqlite' if is_dry_run else 'nct_live.sqlite'
    return Path('data') / name
