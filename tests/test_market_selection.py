"""Tests for market selector, correlation limits, daily notional cap, and volatility circuit breaker."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from nct.config import BudgetConfig, RiskConfig
from nct.exchange.models import Ticker
from nct.risk.protections import LockResult, VolatilityCircuitBreaker
from nct.strategy.market_selector import MarketSelector


def _ticker(inst_id: str, *, volume: float = 1_000_000, spread: float = 0.001) -> Ticker:
    mid = Decimal('65000')
    half_spread = mid * Decimal(str(spread)) / 2
    return Ticker(
        inst_id=inst_id,
        last=mid,
        bid=mid - half_spread,
        ask=mid + half_spread,
        bid_size=Decimal('1'),
        ask_size=Decimal('1'),
        volume_24h=Decimal(str(volume)),
        timestamp=MagicMock(),
    )


# -- MarketSelector -----------------------------------------------------------

class TestMarketSelector:
    def test_all_pass_no_filters(self):
        ms = MarketSelector()
        tickers = {'BTC-USD': _ticker('BTC-USD'), 'ETH-USD': _ticker('ETH-USD')}
        result = ms.filter_pairs(['BTC-USD', 'ETH-USD'], tickers)
        assert result == ['BTC-USD', 'ETH-USD']

    def test_blacklist_removes_pair(self):
        ms = MarketSelector(blacklist=['SHIB-USD'])
        tickers = {'BTC-USD': _ticker('BTC-USD'), 'SHIB-USD': _ticker('SHIB-USD')}
        result = ms.filter_pairs(['BTC-USD', 'SHIB-USD'], tickers)
        assert result == ['BTC-USD']

    def test_low_volume_filtered(self):
        ms = MarketSelector(min_volume_usdt=Decimal('500000'))
        tickers = {
            'BTC-USD': _ticker('BTC-USD', volume=1_000_000),
            'DOGE-USD': _ticker('DOGE-USD', volume=100_000),
        }
        result = ms.filter_pairs(['BTC-USD', 'DOGE-USD'], tickers)
        assert result == ['BTC-USD']

    def test_wide_spread_filtered(self):
        ms = MarketSelector(max_spread_pct=Decimal('0.5'))
        tickers = {
            'BTC-USD': _ticker('BTC-USD', spread=0.001),   # 0.1% - OK
            'ALT-USD': _ticker('ALT-USD', spread=0.01),    # 1.0% - too wide
        }
        result = ms.filter_pairs(['BTC-USD', 'ALT-USD'], tickers)
        assert result == ['BTC-USD']

    def test_missing_ticker_skipped(self):
        ms = MarketSelector()
        tickers = {'BTC-USD': _ticker('BTC-USD')}
        result = ms.filter_pairs(['BTC-USD', 'ETH-USD'], tickers)
        assert result == ['BTC-USD']


# -- Correlation limits (in RiskManager) --------------------------------------

class TestCorrelationLimits:
    def test_correlation_blocks_third_position(self):
        from nct.config import TradingConfig
        from nct.risk.budget_manager import BudgetManager
        from nct.risk.position_sizer import PositionSizer
        from nct.risk.protections import ProtectionManager
        from nct.risk.risk_manager import RiskManager

        risk_config = RiskConfig(
            correlation_groups={'btc_beta': ['BTC-USD', 'ETH-USD', 'SOL-USD']},
            max_correlated_positions=2,
        )
        portfolio = MagicMock()
        portfolio.open_trades = {'BTC-USD': MagicMock(), 'ETH-USD': MagicMock()}
        portfolio.open_trade_count = 2

        rm = RiskManager(
            budget_manager=MagicMock(spec=BudgetManager),
            position_sizer=MagicMock(spec=PositionSizer),
            protection_manager=MagicMock(spec=ProtectionManager),
            risk_config=risk_config,
            trading_config=TradingConfig(max_open_positions=5),
            portfolio=portfolio,
        )
        rm._protections.check.return_value = LockResult(locked=False)

        decision = rm.evaluate_trade(
            inst_id='SOL-USD', side='buy',
            current_price=Decimal('150'), available_balance=Decimal('500'),
            signal_confidence=0.8, open_position_count=2,
        )
        assert decision.approved is False
        assert 'Correlation limit' in decision.denial_reason

    def test_correlation_allows_different_group(self):
        from nct.config import TradingConfig
        from nct.risk.budget_manager import BudgetManager
        from nct.risk.position_sizer import PositionSizer
        from nct.risk.protections import ProtectionManager
        from nct.risk.risk_manager import RiskManager

        risk_config = RiskConfig(
            correlation_groups={'btc_beta': ['BTC-USD', 'ETH-USD']},
            max_correlated_positions=2,
        )
        portfolio = MagicMock()
        portfolio.open_trades = {'BTC-USD': MagicMock()}
        portfolio.open_trade_count = 1

        rm = RiskManager(
            budget_manager=MagicMock(spec=BudgetManager),
            position_sizer=MagicMock(spec=PositionSizer),
            protection_manager=MagicMock(spec=ProtectionManager),
            risk_config=risk_config,
            trading_config=TradingConfig(max_open_positions=5),
            portfolio=portfolio,
        )
        rm._protections.check.return_value = LockResult(locked=False)
        rm._sizer.calculate_size.return_value = Decimal('1')
        rm._budget.can_open_trade.return_value = (True, '')
        rm._budget.budget_remaining = Decimal('400')
        rm._sizer.calculate_stop_loss_price.return_value = Decimal('145')
        rm._sizer.calculate_take_profit_price.return_value = Decimal('160')

        # AVAX is not in btc_beta group, should pass
        decision = rm.evaluate_trade(
            inst_id='AVAX-USD', side='buy',
            current_price=Decimal('35'), available_balance=Decimal('500'),
            signal_confidence=0.8, open_position_count=1,
        )
        assert decision.approved is True

    def test_no_groups_no_block(self):
        from nct.risk.risk_manager import RiskManager

        rm = RiskManager(
            budget_manager=MagicMock(),
            position_sizer=MagicMock(),
            protection_manager=MagicMock(),
            risk_config=RiskConfig(),  # no correlation groups
            trading_config=MagicMock(),
        )
        assert rm._check_correlation('BTC-USD') == ''


# -- Daily notional cap -------------------------------------------------------

class TestDailyNotionalCap:
    def test_cap_blocks_when_exceeded(self):
        from nct.risk.budget_manager import BudgetManager
        from nct.db import Database

        config = BudgetConfig(
            amount_usdt=Decimal('5000'),
            daily_notional_cap_usdt=Decimal('2000'),
        )
        bm = BudgetManager(config, MagicMock(spec=Database))
        bm._period_id = 1
        bm._daily_notional = Decimal('1500')

        allowed, reason = bm.can_open_trade(Decimal('600'))
        assert allowed is False
        assert 'Daily notional cap' in reason

    def test_cap_allows_when_under(self):
        from nct.risk.budget_manager import BudgetManager
        from nct.db import Database

        config = BudgetConfig(
            amount_usdt=Decimal('5000'),
            daily_notional_cap_usdt=Decimal('2000'),
        )
        bm = BudgetManager(config, MagicMock(spec=Database))
        bm._period_id = 1
        bm._daily_notional = Decimal('1000')

        allowed, _ = bm.can_open_trade(Decimal('500'))
        assert allowed is True

    def test_cap_disabled_when_zero(self):
        from nct.risk.budget_manager import BudgetManager
        from nct.db import Database

        config = BudgetConfig(
            amount_usdt=Decimal('5000'),
            max_position_pct=Decimal('100'),
            daily_notional_cap_usdt=Decimal('0'),
        )
        bm = BudgetManager(config, MagicMock(spec=Database))
        bm._period_id = 1

        allowed, _ = bm.can_open_trade(Decimal('4000'))
        assert allowed is True


# -- Volatility circuit breaker -----------------------------------------------

class TestVolatilityCircuitBreaker:
    def test_trips_on_high_volatility(self):
        cb = VolatilityCircuitBreaker(multiplier=2.0, lookback_candles=10)
        # Normal ATR around 100, then spike to 300
        atr_values = [100.0] * 10 + [300.0]
        cb.update_volatility(atr_values)

        result = cb.check(pair='BTC-USD')
        assert result.locked is True
        assert 'VolatilityCircuitBreaker' in result.reason

    def test_no_trip_normal_volatility(self):
        cb = VolatilityCircuitBreaker(multiplier=3.0, lookback_candles=10)
        atr_values = [100.0] * 10 + [120.0]
        cb.update_volatility(atr_values)

        result = cb.check(pair='BTC-USD')
        assert result.locked is False

    def test_resets_when_volatility_normalizes(self):
        cb = VolatilityCircuitBreaker(multiplier=2.0, lookback_candles=10)
        # Trip it
        cb.update_volatility([100.0] * 10 + [300.0])
        assert cb.check(pair='BTC-USD').locked is True

        # Normalize
        cb.update_volatility([100.0] * 11)
        assert cb.check(pair='BTC-USD').locked is False

    def test_insufficient_data_not_tripped(self):
        cb = VolatilityCircuitBreaker(multiplier=2.0, lookback_candles=20)
        cb.update_volatility([100.0] * 5)  # too few
        assert cb.check(pair='BTC-USD').locked is False

    def test_reset_clears_state(self):
        cb = VolatilityCircuitBreaker(multiplier=2.0, lookback_candles=10)
        cb.update_volatility([100.0] * 10 + [300.0])
        assert cb.check(pair='BTC-USD').locked is True

        cb.reset()
        assert cb.check(pair='BTC-USD').locked is False
