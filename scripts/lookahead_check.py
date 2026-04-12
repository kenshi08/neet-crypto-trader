#!/usr/bin/env python3
"""Lookahead bias checker — detects common forward-looking mistakes in strategies.

Usage:
    python scripts/lookahead_check.py --strategy momentum
    python scripts/lookahead_check.py --strategy mean_reversion

Checks that entry signals only depend on prior candle data (shift(1) or earlier),
not the current candle's close/high/low. This is the most common source of
unrealistically good backtests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import pandas as pd

from nct.strategy.factory import create_strategy


def _generate_test_data(n: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a known spike pattern."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='15min'),
        'open': close - np.random.uniform(0.1, 0.5, n),
        'high': close + np.random.uniform(0.5, 2.0, n),
        'low': close - np.random.uniform(0.5, 2.0, n),
        'close': close,
        'volume': np.random.uniform(100, 10000, n),
    })
    return df


def check_lookahead(strategy_name: str, params: dict | None = None) -> list[str]:
    """Check a strategy for potential lookahead bias.

    Returns a list of warning messages. Empty list = no issues detected.
    """
    warnings: list[str] = []
    strategy = create_strategy(strategy_name, params or {})

    df = _generate_test_data(200)
    metadata = {'pair': 'TEST-USD', 'timeframe': '15m'}

    # Run the strategy pipeline
    processed = df.copy()
    processed = strategy.populate_indicators(processed, metadata)
    processed = strategy.populate_entry_trend(processed, metadata)

    # Check 1: Entry signals should reference prior candle (shift(1))
    # Create a modified df where we change only the LAST candle's close
    # If the signal for candle N-1 changes when candle N changes, there's lookahead
    df_modified = df.copy()
    df_modified.iloc[-1, df_modified.columns.get_loc('close')] *= 1.5
    df_modified.iloc[-1, df_modified.columns.get_loc('high')] *= 1.5

    processed_mod = df_modified.copy()
    processed_mod = strategy.populate_indicators(processed_mod, metadata)
    processed_mod = strategy.populate_entry_trend(processed_mod, metadata)

    # Compare signal for second-to-last candle (should be identical)
    idx = -2
    if len(processed) > abs(idx):
        for col in ['enter_long', 'enter_short', 'signal_confidence']:
            if col in processed.columns:
                orig = processed.iloc[idx].get(col, 0)
                mod = processed_mod.iloc[idx].get(col, 0)
                if not np.isclose(float(orig or 0), float(mod or 0), atol=1e-6):
                    warnings.append(
                        f'LOOKAHEAD: Changing candle[-1] affected signal at candle[-2] '
                        f'column={col}: {orig} -> {mod}'
                    )

    # Check 2: Verify signals don't appear on the very first available candle
    # (indicator warmup should prevent this)
    warmup = strategy.required_candle_count
    early_signals = processed.iloc[:warmup]
    early_entries = (
        early_signals.get('enter_long', pd.Series(dtype=float)).sum()
        + early_signals.get('enter_short', pd.Series(dtype=float)).sum()
    )
    if early_entries > 0:
        warnings.append(
            f'WARMUP: {int(early_entries)} entry signals fired during indicator warmup '
            f'(first {warmup} candles) — indicators may use insufficient data'
        )

    # Check 3: Verify that entry signal at row i doesn't use close[i]
    # by checking if signals use shift(1) pattern
    # We modify close at specific indices and check if signals at those indices change
    test_indices = [100, 120, 140, 160]
    for test_idx in test_indices:
        if test_idx >= len(df):
            continue

        df_spike = df.copy()
        df_spike.iloc[test_idx, df_spike.columns.get_loc('close')] *= 2.0
        df_spike.iloc[test_idx, df_spike.columns.get_loc('high')] *= 2.0

        proc_spike = df_spike.copy()
        proc_spike = strategy.populate_indicators(proc_spike, metadata)
        proc_spike = strategy.populate_entry_trend(proc_spike, metadata)

        for col in ['enter_long', 'enter_short']:
            if col not in processed.columns:
                continue
            orig = float(processed.iloc[test_idx].get(col, 0) or 0)
            spiked = float(proc_spike.iloc[test_idx].get(col, 0) or 0)
            if orig != spiked:
                warnings.append(
                    f'SAME-CANDLE: Signal at candle[{test_idx}] changed when '
                    f'close[{test_idx}] was modified — column={col}. '
                    f'Entry may depend on current candle close (not shift(1)).'
                )
                break  # One example is enough

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description='Check strategy for lookahead bias')
    parser.add_argument('--strategy', default='momentum', help='Strategy name')
    args = parser.parse_args()

    print(f'\nChecking strategy "{args.strategy}" for lookahead bias...\n')

    warnings = check_lookahead(args.strategy)

    if not warnings:
        print('  [PASS] No lookahead bias detected.\n')
    else:
        print(f'  [WARN] {len(warnings)} potential issue(s) found:\n')
        for w in warnings:
            print(f'    - {w}')
        print()

    return len(warnings)


if __name__ == '__main__':
    sys.exit(main())
