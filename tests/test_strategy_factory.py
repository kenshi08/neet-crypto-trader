"""Tests for the strategy factory."""

from __future__ import annotations

import pytest

from nct.strategy.factory import create_strategy, list_strategies
from nct.strategy.mean_reversion import MeanReversionStrategy
from nct.strategy.momentum import MomentumStrategy


class TestListStrategies:
    def test_returns_registered_strategies(self):
        strategies = list_strategies()
        assert 'momentum' in strategies
        assert 'mean_reversion' in strategies

    def test_returns_sorted(self):
        strategies = list_strategies()
        assert strategies == sorted(strategies)


class TestCreateStrategy:
    def test_creates_momentum(self):
        strategy = create_strategy('momentum')
        assert isinstance(strategy, MomentumStrategy)
        assert strategy.name == 'momentum'

    def test_creates_mean_reversion(self):
        strategy = create_strategy('mean_reversion')
        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.name == 'mean_reversion'

    def test_creates_momentum_with_params(self):
        strategy = create_strategy(
            'momentum',
            {'rsi_period': 7, 'rsi_oversold': 25, 'rsi_overbought': 75},
        )
        assert isinstance(strategy, MomentumStrategy)
        assert strategy._rsi_period == 7
        assert strategy._rsi_oversold == 25
        assert strategy._rsi_overbought == 75

    def test_creates_mean_reversion_with_params(self):
        strategy = create_strategy(
            'mean_reversion',
            {'bb_period': 30, 'bb_std': 2.5},
        )
        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy._bb_period == 30
        assert strategy._bb_std == 2.5

    def test_empty_params_uses_defaults(self):
        strategy = create_strategy('momentum', {})
        assert isinstance(strategy, MomentumStrategy)
        # Default RSI period is 14
        assert strategy._rsi_period == 14

    def test_none_params_uses_defaults(self):
        strategy = create_strategy('momentum', None)
        assert isinstance(strategy, MomentumStrategy)
        assert strategy._rsi_period == 14

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match='Unknown strategy'):
            create_strategy('nonexistent_strategy')

    def test_unknown_strategy_lists_available(self):
        with pytest.raises(ValueError, match='Available strategies'):
            create_strategy('fake')
