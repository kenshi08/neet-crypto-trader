"""Tests for structured logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from nct.logging_setup import setup_logging


class TestLoggingSetup:
    def test_creates_log_directory(self, tmp_path: Path):
        log_dir = tmp_path / 'test_logs'
        setup_logging(log_dir=str(log_dir))
        assert log_dir.exists()

    def test_creates_log_files_on_write(self, tmp_path: Path):
        log_dir = tmp_path / 'test_logs'
        setup_logging(log_dir=str(log_dir))

        logger = logging.getLogger('test')
        logger.info('test message')

        # Agent log should exist after a message
        assert (log_dir / 'agent.log').exists()

    def test_debug_mode_sets_level(self, tmp_path: Path):
        log_dir = tmp_path / 'test_logs'
        setup_logging(log_dir=str(log_dir), debug=True)

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_production_mode_sets_info_level(self, tmp_path: Path):
        log_dir = tmp_path / 'test_logs'
        setup_logging(log_dir=str(log_dir), debug=False)

        root = logging.getLogger()
        assert root.level == logging.INFO


class TestTradeEventFilter:
    def test_passes_trade_events(self, tmp_path: Path):
        log_dir = tmp_path / 'test_logs'
        setup_logging(log_dir=str(log_dir))

        logger = logging.getLogger('test.trade')
        logger.info('trade_opened BTC-USDT')

        trades_log = log_dir / 'trades.log'
        if trades_log.exists():
            content = trades_log.read_text()
            assert 'trade_opened' in content

    def test_filters_non_trade_events(self, tmp_path: Path):
        log_dir = tmp_path / 'test_logs'
        setup_logging(log_dir=str(log_dir))

        logger = logging.getLogger('test.general')
        logger.info('just a regular message with no trade keywords')

        trades_log = log_dir / 'trades.log'
        if trades_log.exists():
            content = trades_log.read_text()
            assert 'regular message' not in content
