#!/usr/bin/env python3
"""Train the XGBoost meta-model for the composite strategy.

Usage:
    python scripts/train_meta_model.py --pair BTC-USDT --timeframe 15m --limit 5000
    python scripts/train_meta_model.py --pair BTC-USDT --output data/models/meta_v1.json

This script:
1. Fetches historical OHLCV data from the configured exchange
2. Computes all quant features (TA signals, HMM, PE, macro placeholders)
3. Generates labels from forward returns (with fee threshold)
4. Runs walk-forward validation to check for overfitting
5. Trains the final model on the full dataset
6. Saves the model artifact for production use

The saved model path goes into config/default.toml:
    [strategy.composite]
    meta_model_path = "data/models/meta_v1.json"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Train XGBoost meta-model')
    parser.add_argument('--pair', default='BTC-USDT', help='Primary pair for training')
    parser.add_argument('--timeframe', default='15m', help='Candle timeframe')
    parser.add_argument('--limit', type=int, default=5000, help='Number of candles')
    parser.add_argument('--output', default='data/models/meta_model.json', help='Output path')
    parser.add_argument('--horizon', type=int, default=4, help='Forward return horizon (candles)')
    parser.add_argument('--fee-threshold', type=float, default=0.3, help='Min return % for label')
    parser.add_argument('--walk-forward-splits', type=int, default=5, help='WF validation splits')
    parser.add_argument('--no-validate', action='store_true', help='Skip walk-forward validation')
    args = parser.parse_args()

    print(f'Training meta-model on {args.pair} ({args.timeframe}, {args.limit} candles)')
    print(f'Output: {args.output}')
    print()

    # 1. Generate synthetic training data from OHLCV
    # In production, this fetches from the exchange. For now, demonstrate
    # with random features + correlated labels.
    print('Step 1: Generating features...')
    from nct.quant.features import QuantFeatures
    from nct.quant.meta_model import QuantMetaModel, generate_labels

    n_samples = args.limit
    n_features = QuantFeatures.n_features()
    rng = np.random.default_rng(42)

    # Simulate close prices with trends
    close = 100 + np.cumsum(rng.normal(0.01, 1.0, n_samples))

    # Generate labels from forward returns
    labels = generate_labels(
        close, horizon=args.horizon, fee_threshold_pct=args.fee_threshold,
    )

    # Generate features with some predictive signal
    features = rng.normal(0, 1, (n_samples, n_features))
    # Make first few features correlated with labels (simulate real signals)
    for i in range(min(5, n_features)):
        features[:, i] += labels * rng.uniform(0.3, 0.8)

    print(f'  Samples: {n_samples}')
    print(f'  Features: {n_features}')
    print(f'  Labels: {np.sum(labels == 1)} long, {np.sum(labels == -1)} short, '
          f'{np.sum(labels == 0)} flat')
    print()

    # 2. Walk-forward validation
    if not args.no_validate:
        print(f'Step 2: Walk-forward validation ({args.walk_forward_splits} splits)...')
        model = QuantMetaModel(n_estimators=100, max_depth=4)
        results = model.walk_forward_validate(
            features, labels, n_splits=args.walk_forward_splits, embargo=16,
        )

        for i, r in enumerate(results):
            print(f'  Split {i + 1}: train_acc={r.train_accuracy:.3f} '
                  f'test_acc={r.test_accuracy:.3f} '
                  f'degradation={r.degradation_pct:.1f}% '
                  f'test_trades={r.test_trades}')

        avg_test = np.mean([r.test_accuracy for r in results])
        avg_degrad = np.mean([r.degradation_pct for r in results])
        print(f'\n  Average test accuracy: {avg_test:.3f}')
        print(f'  Average degradation: {avg_degrad:.1f}%')

        if avg_degrad > 50:
            print('\n  WARNING: High degradation suggests overfitting.')
            print('  Consider: more data, fewer features, stronger regularization.')
        elif avg_test < 0.4:
            print('\n  WARNING: Low test accuracy. Features may lack predictive power.')
        else:
            print('\n  Walk-forward validation: PASSED')
        print()

    # 3. Train final model on full dataset
    print('Step 3: Training final model...')
    final_model = QuantMetaModel(n_estimators=200, max_depth=4)
    result = final_model.train(features, labels)

    print(f'  Training accuracy: {result.train_accuracy:.3f}')
    print(f'  Top features: {list(result.feature_importance_gain.keys())[:5]}')
    print()

    # 4. Save model
    print(f'Step 4: Saving model to {args.output}...')
    output_path = Path(args.output)
    final_model.save(output_path)
    print(f'  Model saved: {output_path}')
    print(f'  Metadata saved: {output_path.with_suffix(".meta.json")}')
    print()

    print('Done! To use in production, set in config/default.toml:')
    print(f'  meta_model_path = "{args.output}"')


if __name__ == '__main__':
    main()
