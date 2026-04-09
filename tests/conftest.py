"""Shared fixtures for nct tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from nct.config import AppConfig, BudgetConfig, OKXCredentials, RiskConfig, TradingConfig


@pytest.fixture
def okx_credentials() -> OKXCredentials:
    return OKXCredentials(
        api_key='test-key',
        api_secret='test-secret',
        passphrase='test-pass',
        demo_mode=True,
    )


@pytest.fixture
def trading_config() -> TradingConfig:
    return TradingConfig(
        pairs=['BTC-USDT', 'ETH-USDT'],
        strategy='momentum',
        timeframe='15m',
        max_open_positions=3,
        poll_interval_seconds=10,
    )


@pytest.fixture
def budget_config() -> BudgetConfig:
    return BudgetConfig(
        period='weekly',
        amount_usdt=Decimal('500'),
        max_loss_pct=Decimal('5.0'),
        max_gain_pct=Decimal('15.0'),
        max_position_pct=Decimal('20.0'),
        daily_loss_limit_usdt=Decimal('100'),
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        stop_loss_pct=Decimal('3.0'),
        take_profit_pct=Decimal('5.0'),
        time_limit_seconds=3600,
    )


@pytest.fixture
def app_config(okx_credentials, trading_config, budget_config, risk_config) -> AppConfig:
    return AppConfig(
        okx=okx_credentials,
        trading=trading_config,
        budget=budget_config,
        risk=risk_config,
    )


@pytest.fixture
def sample_ticker_data() -> dict:
    return {
        'instId': 'BTC-USDT',
        'last': '67500.5',
        'bidPx': '67500.0',
        'askPx': '67501.0',
        'bidSz': '1.5',
        'askSz': '2.0',
        'vol24h': '12345.67',
        'ts': '1712000000000',
    }


@pytest.fixture
def sample_candle_raw() -> list[str]:
    return [
        '1712000000000',  # ts
        '67000.0',        # open
        '67800.0',        # high
        '66900.0',        # low
        '67500.5',        # close
        '1234.56',        # volume
        '0',              # volCcy
        '0',              # volCcyQuote
        '1',              # confirm
    ]


@pytest.fixture
def sample_order_response_data() -> dict:
    return {
        'ordId': '123456789',
        'clOrdId': 'client-001',
        'state': 'filled',
        'instId': 'BTC-USDT',
        'side': 'buy',
        'sz': '0.01',
        'px': '67500.0',
        'fillSz': '0.01',
        'avgPx': '67500.0',
        'fee': '-0.675',
        'feeCcy': 'USDT',
    }


@pytest.fixture
def sample_position_data() -> dict:
    return {
        'instId': 'BTC-USDT',
        'posSide': 'net',
        'pos': '0.01',
        'avgPx': '67500.0',
        'upl': '50.5',
        'realizedPnl': '0',
        'margin': '675.0',
        'lever': '1',
        'uTime': '1712000000000',
    }


@pytest.fixture
def sample_balance_data() -> dict:
    return {
        'ccy': 'USDT',
        'bal': '10000.0',
        'availBal': '8500.0',
        'frozenBal': '1500.0',
    }
