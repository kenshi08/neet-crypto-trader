#!/usr/bin/env python3
"""Automated pipeline — walk-forward validation + conditional retrain.

Runs on a schedule (weekly cron or Docker). For each run:
1. Fetches recent OHLCV data (or uses synthetic for now)
2. Computes quant features
3. Runs walk-forward validation on the current meta-model
4. If performance degrades below threshold, retrains automatically
5. Saves results to SQLite pipeline_runs table
6. Optionally sends Telegram notification with summary

Usage:
    # One-shot run
    python scripts/auto_pipeline.py

    # With custom thresholds
    python scripts/auto_pipeline.py --min-accuracy 0.55 --max-degradation 40

    # Retrain only (skip validation)
    python scripts/auto_pipeline.py --retrain-only

    # Docker cron (add to crontab):
    # 0 2 * * 0  python /app/scripts/auto_pipeline.py >> /app/logs/pipeline.log 2>&1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


async def run_pipeline(args: argparse.Namespace) -> dict:
    """Execute the full pipeline: validate + conditional retrain."""
    from nct.db import Database, get_db_path
    from nct.quant.features import QuantFeatures
    from nct.quant.meta_model import QuantMetaModel, generate_labels

    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(UTC).isoformat()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    current_model_path = model_dir / 'meta_model.pkl'

    print(f'=== Pipeline Run {run_id} @ {timestamp} ===')
    print()

    # --- Step 1: Generate features + labels ---
    print('Step 1: Preparing data...')
    n_samples = args.n_samples
    n_features = QuantFeatures.n_features()
    rng = np.random.default_rng(int(datetime.now(UTC).timestamp()) % 2**31)

    # Simulate prices with regime shifts
    close = np.zeros(n_samples)
    close[0] = 100.0
    for i in range(1, n_samples):
        # Regime shifts every ~500 candles
        trend = 0.02 if (i // 500) % 2 == 0 else -0.01
        close[i] = close[i - 1] + rng.normal(trend, 0.8)

    labels = generate_labels(
        close, horizon=args.horizon, fee_threshold_pct=args.fee_threshold,
    )

    features = rng.normal(0, 1, (n_samples, n_features))
    for i in range(min(8, n_features)):
        features[:, i] += labels * rng.uniform(0.2, 0.6)

    label_dist = {
        'long': int(np.sum(labels == 1)),
        'short': int(np.sum(labels == -1)),
        'flat': int(np.sum(labels == 0)),
    }
    print(f'  Samples: {n_samples}, Labels: {label_dist}')

    # --- Step 2: Walk-forward validation ---
    print(f'\nStep 2: Walk-forward validation ({args.wf_splits} splits, '
          f'embargo={args.embargo})...')

    model = QuantMetaModel(n_estimators=150, max_depth=4)
    wf_results = model.walk_forward_validate(
        features, labels, n_splits=args.wf_splits, embargo=args.embargo,
    )

    for i, r in enumerate(wf_results):
        status = 'OK' if r.degradation_pct < args.max_degradation else 'WARN'
        print(f'  Split {i + 1}: test_acc={r.test_accuracy:.3f} '
              f'degrad={r.degradation_pct:.1f}% [{status}]')

    avg_accuracy = np.mean([r.test_accuracy for r in wf_results])
    avg_degradation = np.mean([r.degradation_pct for r in wf_results])
    print(f'\n  Average: accuracy={avg_accuracy:.3f}, '
          f'degradation={avg_degradation:.1f}%')

    # --- Step 3: Decide whether to retrain ---
    needs_retrain = (
        args.retrain_only
        or avg_accuracy < args.min_accuracy
        or avg_degradation > args.max_degradation
        or not current_model_path.exists()
    )

    retrain_reason = 'scheduled' if args.retrain_only else ''
    if avg_accuracy < args.min_accuracy:
        retrain_reason = f'accuracy {avg_accuracy:.3f} < {args.min_accuracy}'
    if avg_degradation > args.max_degradation:
        retrain_reason = f'degradation {avg_degradation:.1f}% > {args.max_degradation}%'
    if not current_model_path.exists():
        retrain_reason = 'no existing model'

    train_accuracy = 0.0
    model_path_str = ''

    if needs_retrain:
        print(f'\nStep 3: RETRAINING (reason: {retrain_reason})...')
        final_model = QuantMetaModel(n_estimators=200, max_depth=4)
        result = final_model.train(features, labels)
        train_accuracy = result.train_accuracy

        # Save with version
        version = datetime.now(UTC).strftime('%Y%m%d_%H%M')
        versioned_path = model_dir / f'meta_model_v{version}.pkl'
        final_model.save(versioned_path)

        # Also save as current
        final_model.save(current_model_path)
        model_path_str = str(current_model_path)

        print(f'  Train accuracy: {train_accuracy:.3f}')
        print(f'  Saved: {versioned_path}')
        print(f'  Current: {current_model_path}')
    else:
        print('\nStep 3: No retrain needed (performance within thresholds)')

    # --- Step 4: Save results to DB ---
    print('\nStep 4: Saving results...')
    try:
        db_path = get_db_path(is_dry_run=True)
        db = Database(db_path)
        await db.connect()
        await db.conn.execute(
            """INSERT INTO pipeline_runs
               (run_id, timestamp, run_type, n_samples,
                walk_forward_accuracy, degradation_pct,
                train_accuracy, model_path, status, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, timestamp,
                'retrain' if needs_retrain else 'validate',
                n_samples, avg_accuracy, avg_degradation,
                train_accuracy, model_path_str,
                'retrained' if needs_retrain else 'passed',
                json.dumps({
                    'label_dist': label_dist,
                    'retrain_reason': retrain_reason,
                    'wf_splits': [
                        {
                            'test_accuracy': r.test_accuracy,
                            'degradation_pct': r.degradation_pct,
                            'test_trades': r.test_trades,
                        }
                        for r in wf_results
                    ],
                }),
            ),
        )
        await db.conn.commit()
        await db.close()
        print(f'  Saved to pipeline_runs (run_id={run_id})')
    except Exception as e:
        print(f'  DB save failed: {e}')

    # --- Summary ---
    summary = {
        'run_id': run_id,
        'timestamp': timestamp,
        'n_samples': n_samples,
        'avg_accuracy': round(avg_accuracy, 4),
        'avg_degradation': round(avg_degradation, 1),
        'retrained': needs_retrain,
        'retrain_reason': retrain_reason,
        'train_accuracy': round(train_accuracy, 4),
        'model_path': model_path_str,
    }

    print(f'\n=== Pipeline Complete ===')
    action = 'RETRAINED' if needs_retrain else 'VALIDATED'
    print(f'Result: {action} | Accuracy: {avg_accuracy:.3f} | '
          f'Degradation: {avg_degradation:.1f}%')

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Automated validation + retrain pipeline')
    parser.add_argument('--n-samples', type=int, default=3000, help='Training samples')
    parser.add_argument('--horizon', type=int, default=4, help='Forward return horizon')
    parser.add_argument('--fee-threshold', type=float, default=0.3, help='Min return % for label')
    parser.add_argument('--wf-splits', type=int, default=5, help='Walk-forward splits')
    parser.add_argument('--embargo', type=int, default=16, help='Embargo candles between train/test')
    parser.add_argument('--min-accuracy', type=float, default=0.52, help='Min OOS accuracy')
    parser.add_argument('--max-degradation', type=float, default=40.0, help='Max degradation %')
    parser.add_argument('--model-dir', default='data/models', help='Model artifact directory')
    parser.add_argument('--retrain-only', action='store_true', help='Force retrain')
    args = parser.parse_args()

    asyncio.run(run_pipeline(args))


if __name__ == '__main__':
    main()
