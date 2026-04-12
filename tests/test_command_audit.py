"""Tests for Telegram command audit trail in the database."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nct.db import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / 'test.sqlite')
    await database.connect()
    yield database
    await database.close()


class TestTelegramCommandLog:
    @pytest.mark.asyncio
    async def test_log_and_query(self, db):
        await db.log_telegram_command(
            user_id='12345',
            command='status',
            args='',
            result='success',
        )
        cursor = await db.conn.execute(
            "SELECT * FROM telegram_commands WHERE command = 'status'"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row['user_id'] == '12345'
        assert row['command'] == 'status'
        assert row['result'] == 'success'

    @pytest.mark.asyncio
    async def test_log_with_args(self, db):
        await db.log_telegram_command(
            user_id='12345',
            command='why',
            args='BTC-USD',
            result='success',
        )
        cursor = await db.conn.execute(
            "SELECT * FROM telegram_commands WHERE command = 'why'"
        )
        row = dict((await cursor.fetchone()))
        assert row['args'] == 'BTC-USD'

    @pytest.mark.asyncio
    async def test_log_error_result(self, db):
        await db.log_telegram_command(
            user_id='12345',
            command='close',
            args='ETH-USD',
            result='error',
        )
        cursor = await db.conn.execute(
            "SELECT * FROM telegram_commands WHERE result = 'error'"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_log_with_explicit_timestamp(self, db):
        ts = datetime(2026, 4, 12, 10, 0, 0, tzinfo=UTC)
        await db.log_telegram_command(
            user_id='12345',
            command='balance',
            timestamp=ts,
        )
        cursor = await db.conn.execute(
            "SELECT timestamp FROM telegram_commands WHERE command = 'balance'"
        )
        row = dict((await cursor.fetchone()))
        assert '2026-04-12' in row['timestamp']

    @pytest.mark.asyncio
    async def test_multiple_commands_ordered(self, db):
        for cmd in ['status', 'why', 'close', 'balance']:
            await db.log_telegram_command(user_id='99', command=cmd)

        cursor = await db.conn.execute(
            "SELECT command FROM telegram_commands ORDER BY id"
        )
        rows = await cursor.fetchall()
        commands = [dict(r)['command'] for r in rows]
        assert commands == ['status', 'why', 'close', 'balance']
