"""Structured logging configuration — structlog with file and console output."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


def setup_logging(*, log_dir: str = 'logs', debug: bool = False) -> None:
    """Configure structlog with console + file output.

    Log files:
    - agent.log: all INFO+ messages
    - trades.log: trade events only (filtered by 'trade_' prefix in event)
    - errors.log: WARNING+ messages

    Args:
        log_dir: directory for log files
        debug: if True, set log level to DEBUG and use verbose console output
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if debug else logging.INFO

    # -- Standard library logging (for file output) --------------------

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    # Console handler — always present
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    root.addHandler(console)

    # File handlers — gracefully skip if directory isn't writable
    try:
        # Agent log — all INFO+
        agent_handler = RotatingFileHandler(
            log_path / 'agent.log',
            maxBytes=10_000_000,  # 10 MB
            backupCount=7,
            encoding='utf-8',
        )
        agent_handler.setLevel(logging.INFO)
        root.addHandler(agent_handler)

        # Error log — WARNING+
        error_handler = RotatingFileHandler(
            log_path / 'errors.log',
            maxBytes=10_000_000,
            backupCount=7,
            encoding='utf-8',
        )
        error_handler.setLevel(logging.WARNING)
        root.addHandler(error_handler)

        # Trade log — INFO, filtered to trade events only
        trade_handler = RotatingFileHandler(
            log_path / 'trades.log',
            maxBytes=10_000_000,
            backupCount=7,
            encoding='utf-8',
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.addFilter(_TradeEventFilter())
        root.addHandler(trade_handler)
    except PermissionError:
        print(
            f'WARNING: Cannot write to {log_dir}/ — logging to console only. '
            f'Fix with: chown -R 1000:1000 {log_dir}/',
            file=sys.stderr,
        )

    # -- Structlog configuration ----------------------------------------

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if debug:
        # Development: colored console output
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        )
    else:
        # Production: JSON output for files, console renderer for stderr
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        )


class _TradeEventFilter(logging.Filter):
    """Only pass log records that contain trade-related events."""

    _TRADE_KEYWORDS = (
        'trade_', 'order_', 'position_', 'budget_',
        'stop_loss', 'take_profit', 'kill_switch',
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return any(kw in msg for kw in self._TRADE_KEYWORDS)
