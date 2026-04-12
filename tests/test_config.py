"""Tests for configuration loading and validation."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from nct.config import (
    AppConfig,
    BudgetConfig,
    MarketSelectionConfig,
    OKXCredentials,
    RiskConfig,
    TradingConfig,
    load_config,
    validate_live_config,
)


class TestOKXCredentials:
    def test_demo_flag(self):
        creds = OKXCredentials(demo_mode=True)
        assert creds.flag == '1'

    def test_live_flag(self):
        creds = OKXCredentials(demo_mode=False)
        assert creds.flag == '0'

    def test_defaults_to_demo(self):
        creds = OKXCredentials()
        assert creds.demo_mode is True


class TestTradingConfig:
    def test_valid_pairs(self):
        config = TradingConfig(pairs=['BTC-USDT', 'ETH-USDT'])
        assert config.pairs == ['BTC-USDT', 'ETH-USDT']

    def test_invalid_pair_format(self):
        with pytest.raises(ValueError, match='Invalid pair format'):
            TradingConfig(pairs=['BTCUSDT'])

    def test_valid_timeframe(self):
        config = TradingConfig(timeframe='5m')
        assert config.timeframe == '5m'

    def test_invalid_timeframe(self):
        with pytest.raises(ValueError, match='Invalid timeframe'):
            TradingConfig(timeframe='7m')

    def test_all_valid_timeframes(self):
        valid = ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M']
        for tf in valid:
            config = TradingConfig(timeframe=tf)
            assert config.timeframe == tf


class TestBudgetConfig:
    def test_decimal_coercion_from_int(self):
        config = BudgetConfig(amount_usdt=500)
        assert config.amount_usdt == Decimal('500')
        assert isinstance(config.amount_usdt, Decimal)

    def test_decimal_coercion_from_float(self):
        config = BudgetConfig(max_loss_pct=5.0)
        assert config.max_loss_pct == Decimal('5.0')

    def test_decimal_coercion_from_string(self):
        config = BudgetConfig(amount_usdt='750.50')
        assert config.amount_usdt == Decimal('750.50')

    def test_valid_periods(self):
        weekly = BudgetConfig(period='weekly')
        monthly = BudgetConfig(period='monthly')
        assert weekly.period == 'weekly'
        assert monthly.period == 'monthly'

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            BudgetConfig(period='daily')


class TestRiskConfig:
    def test_defaults(self):
        config = RiskConfig()
        assert config.stop_loss_pct == Decimal('3.0')
        assert config.take_profit_pct == Decimal('5.0')
        assert config.time_limit_seconds == 3600
        assert config.trailing_stop is False
        assert config.min_signal_confidence == 0.6

    def test_decimal_fields(self):
        config = RiskConfig(stop_loss_pct=2.5, take_profit_pct=8)
        assert config.stop_loss_pct == Decimal('2.5')
        assert config.take_profit_pct == Decimal('8')


class TestLoadConfig:
    def test_load_from_default_toml(self, tmp_path: Path):
        toml_content = b"""
[trading]
pairs = ["SOL-USDT"]
strategy = "momentum"
timeframe = "5m"
max_open_positions = 2
poll_interval_seconds = 5

[budget]
period = "monthly"
amount_usdt = 1000
max_loss_pct = 3.0
max_gain_pct = 10.0
max_position_pct = 25.0
daily_loss_limit_usdt = 50

[risk]
stop_loss_pct = 2.0
take_profit_pct = 4.0
time_limit_seconds = 1800
"""
        config_file = tmp_path / 'test_config.toml'
        config_file.write_bytes(toml_content)

        config = load_config(config_file)

        assert config.trading.pairs == ['SOL-USDT']
        assert config.trading.timeframe == '5m'
        assert config.trading.max_open_positions == 2
        assert config.budget.period == 'monthly'
        assert config.budget.amount_usdt == Decimal('1000')
        assert config.risk.stop_loss_pct == Decimal('2.0')

    def test_load_missing_file_uses_defaults(self, tmp_path: Path):
        config = load_config(tmp_path / 'nonexistent.toml')

        assert config.trading.pairs == ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
        assert config.budget.period == 'weekly'
        assert config.okx.demo_mode is True

    def test_partial_toml_merges_with_defaults(self, tmp_path: Path):
        toml_content = b"""
[trading]
pairs = ["DOGE-USDT"]
"""
        config_file = tmp_path / 'partial.toml'
        config_file.write_bytes(toml_content)

        config = load_config(config_file)

        assert config.trading.pairs == ['DOGE-USDT']
        # Other trading fields use defaults
        assert config.trading.timeframe == '15m'
        # Budget and risk sections use full defaults
        assert config.budget.amount_usdt == Decimal('500')
        assert config.risk.stop_loss_pct == Decimal('3.0')


class TestAppConfig:
    def test_full_construction(self, app_config: AppConfig):
        assert app_config.okx.demo_mode is True
        assert app_config.trading.pairs == ['BTC-USDT', 'ETH-USDT']
        assert app_config.budget.amount_usdt == Decimal('500')
        assert app_config.risk.stop_loss_pct == Decimal('3.0')


# ===================================================================
# Issue #88 — Live-mode config validation
# ===================================================================


def _safe_config(**overrides) -> AppConfig:
    """Build a config that passes all live validation checks."""
    risk_kwargs = {
        'stop_loss_pct': Decimal('3.0'),
        'take_profit_pct': Decimal('5.0'),
        'min_signal_confidence': 0.6,
        'volatility_circuit_breaker_multiplier': 3.0,
        'correlation_groups': {'btc_beta': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']},
    }
    risk_kwargs.update(overrides.pop('risk', {}))
    budget_kwargs = {'daily_loss_limit_usdt': Decimal('50')}
    budget_kwargs.update(overrides.pop('budget', {}))
    ms_kwargs = {'min_volume_usdt': Decimal('10000')}
    ms_kwargs.update(overrides.pop('market_selection', {}))
    trading_kwargs = {'pairs': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']}
    trading_kwargs.update(overrides.pop('trading', {}))

    return AppConfig(
        trading=TradingConfig(**trading_kwargs),
        budget=BudgetConfig(**budget_kwargs),
        risk=RiskConfig(**risk_kwargs),
        market_selection=MarketSelectionConfig(**ms_kwargs),
        **overrides,
    )


class TestValidateLiveConfig:
    def test_safe_config_passes(self):
        config = _safe_config()
        errors = validate_live_config(config)
        assert errors == []

    def test_stop_loss_zero_fails(self):
        config = _safe_config(risk={'stop_loss_pct': Decimal('0')})
        errors = validate_live_config(config)
        assert any('stop_loss_pct' in e for e in errors)

    def test_take_profit_zero_fails(self):
        config = _safe_config(risk={'take_profit_pct': Decimal('0')})
        errors = validate_live_config(config)
        assert any('take_profit_pct' in e for e in errors)

    def test_low_confidence_fails(self):
        config = _safe_config(risk={'min_signal_confidence': 0.2})
        errors = validate_live_config(config)
        assert any('min_signal_confidence' in e for e in errors)

    def test_confidence_at_threshold_passes(self):
        config = _safe_config(risk={'min_signal_confidence': 0.5})
        errors = validate_live_config(config)
        assert not any('min_signal_confidence' in e for e in errors)

    def test_daily_loss_limit_zero_fails(self):
        config = _safe_config(budget={'daily_loss_limit_usdt': Decimal('0')})
        errors = validate_live_config(config)
        assert any('daily_loss_limit' in e for e in errors)

    def test_circuit_breaker_disabled_fails(self):
        config = _safe_config(risk={'volatility_circuit_breaker_multiplier': 0.0})
        errors = validate_live_config(config)
        assert any('circuit_breaker' in e.lower() or 'volatility' in e.lower() for e in errors)

    def test_no_correlation_groups_with_many_pairs_fails(self):
        config = _safe_config(
            trading={'pairs': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']},
            risk={'correlation_groups': {}},
        )
        errors = validate_live_config(config)
        assert any('correlation' in e.lower() for e in errors)

    def test_few_pairs_skip_correlation_check(self):
        config = _safe_config(
            trading={'pairs': ['BTC-USDT', 'ETH-USDT']},
            risk={'correlation_groups': {}},
        )
        errors = validate_live_config(config)
        assert not any('correlation' in e.lower() for e in errors)

    def test_no_market_quality_filters_fails(self):
        config = _safe_config(
            market_selection={'min_volume_usdt': Decimal('0'), 'max_spread_pct': Decimal('0')},
        )
        errors = validate_live_config(config)
        assert any('market quality' in e.lower() for e in errors)

    def test_spread_filter_alone_passes(self):
        config = _safe_config(
            market_selection={'min_volume_usdt': Decimal('0'), 'max_spread_pct': Decimal('1.0')},
        )
        errors = validate_live_config(config)
        assert not any('market quality' in e.lower() for e in errors)

    def test_multiple_violations_returns_all(self):
        config = _safe_config(
            risk={
                'stop_loss_pct': Decimal('0'),
                'take_profit_pct': Decimal('0'),
                'min_signal_confidence': 0.1,
                'volatility_circuit_breaker_multiplier': 0.0,
                'correlation_groups': {},
            },
            budget={'daily_loss_limit_usdt': Decimal('0')},
            market_selection={'min_volume_usdt': Decimal('0'), 'max_spread_pct': Decimal('0')},
        )
        errors = validate_live_config(config)
        assert len(errors) >= 6
