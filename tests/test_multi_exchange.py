"""Tests for the multi-exchange portfolio manager."""

from __future__ import annotations

import pytest

from nct.portfolio.multi_exchange import (
    DEFAULT_PORTFOLIO_CONFIGS,
    MultiExchangeManager,
    MultiPortfolioStatus,
    PortfolioConfig,
    PortfolioState,
)

# ---------------------------------------------------------------------------
# Tests: PortfolioConfig
# ---------------------------------------------------------------------------

class TestPortfolioConfig:
    def test_defaults(self):
        cfg = PortfolioConfig(exchange='okx', strategy='momentum')
        assert cfg.budget_allocation_pct == 33.3
        assert cfg.max_positions == 2
        assert cfg.enabled is True

    def test_custom_values(self):
        cfg = PortfolioConfig(
            exchange='hyperliquid',
            strategy='pairs_kalman',
            pairs=['BTC-USDT', 'ETH-USDT'],
            budget_allocation_pct=40.0,
            max_positions=3,
        )
        assert cfg.pairs == ['BTC-USDT', 'ETH-USDT']
        assert cfg.budget_allocation_pct == 40.0


# ---------------------------------------------------------------------------
# Tests: PortfolioState
# ---------------------------------------------------------------------------

class TestPortfolioState:
    def test_win_rate_no_trades(self):
        state = PortfolioState(exchange='okx', strategy='momentum')
        assert state.win_rate == 0.0

    def test_win_rate_with_trades(self):
        state = PortfolioState(
            exchange='okx', strategy='momentum',
            win_count=7, loss_count=3,
        )
        assert state.win_rate == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Tests: MultiExchangeManager
# ---------------------------------------------------------------------------

class TestMultiExchangeManager:
    @pytest.fixture()
    def configs(self):
        return [
            PortfolioConfig(
                exchange='hyperliquid', strategy='pairs_kalman',
                max_positions=2, budget_allocation_pct=40,
            ),
            PortfolioConfig(
                exchange='okx', strategy='composite',
                max_positions=3, budget_allocation_pct=35,
            ),
            PortfolioConfig(
                exchange='coinbase', strategy='composite',
                max_positions=2, budget_allocation_pct=25,
            ),
        ]

    @pytest.fixture()
    def manager(self, configs):
        return MultiExchangeManager(
            portfolio_configs=configs,
            max_total_positions=6,
        )

    def test_active_exchanges_empty_initially(self, manager):
        # Not running yet
        assert manager.active_exchanges == []

    def test_get_status(self, manager):
        status = manager.get_status()
        assert isinstance(status, MultiPortfolioStatus)
        assert len(status.portfolios) == 3
        assert status.total_open_positions == 0

    def test_can_open_position_basic(self, manager):
        allowed, _reason = manager.can_open_position('okx', 'BTC-USDT')
        assert allowed is True

    def test_can_open_unknown_exchange(self, manager):
        allowed, reason = manager.can_open_position('binance', 'BTC-USDT')
        assert allowed is False
        assert 'not configured' in reason

    def test_per_portfolio_position_limit(self, manager):
        # Fill up hyperliquid (max 2)
        manager.record_trade('hyperliquid', pnl=0.0, is_open=True)
        manager.record_trade('hyperliquid', pnl=0.0, is_open=True)

        allowed, reason = manager.can_open_position('hyperliquid', 'SOL-USDT')
        assert allowed is False
        assert 'max positions' in reason

    def test_global_position_limit(self):
        configs = [
            PortfolioConfig(exchange='okx', strategy='x', max_positions=10),
        ]
        mgr = MultiExchangeManager(
            portfolio_configs=configs, max_total_positions=2,
        )
        mgr.record_trade('okx', pnl=0.0, is_open=True)
        mgr.record_trade('okx', pnl=0.0, is_open=True)

        allowed, reason = mgr.can_open_position('okx', 'BTC')
        assert allowed is False
        assert 'Global' in reason

    def test_record_trade_open(self, manager):
        manager.record_trade('okx', pnl=0.0, is_open=True)
        assert manager._states['okx'].open_positions == 1

    def test_record_trade_close_win(self, manager):
        manager.record_trade('okx', pnl=0.0, is_open=True)
        manager.record_trade('okx', pnl=5.0, is_open=False)
        state = manager._states['okx']
        assert state.open_positions == 0
        assert state.win_count == 1
        assert state.realized_pnl == 5.0

    def test_record_trade_close_loss(self, manager):
        manager.record_trade('okx', pnl=0.0, is_open=True)
        manager.record_trade('okx', pnl=-3.0, is_open=False)
        state = manager._states['okx']
        assert state.loss_count == 1
        assert state.realized_pnl == -3.0

    def test_aggregate_win_rate(self, manager):
        manager.record_trade('okx', pnl=0.0, is_open=True)
        manager.record_trade('okx', pnl=5.0, is_open=False)
        manager.record_trade('hyperliquid', pnl=0.0, is_open=True)
        manager.record_trade('hyperliquid', pnl=-2.0, is_open=False)

        status = manager.get_status()
        assert status.aggregate_win_rate == pytest.approx(0.5)
        assert status.total_realized_pnl == pytest.approx(3.0)

    def test_budget_allocation(self, manager):
        total = 1000.0
        assert manager.get_budget_allocation('hyperliquid', total) == 400.0
        assert manager.get_budget_allocation('okx', total) == 350.0
        assert manager.get_budget_allocation('coinbase', total) == 250.0

    def test_budget_allocation_unknown_exchange(self, manager):
        assert manager.get_budget_allocation('binance', 1000.0) == 0.0

    def test_disabled_portfolio_excluded(self):
        configs = [
            PortfolioConfig(exchange='okx', strategy='x', enabled=True),
            PortfolioConfig(exchange='coinbase', strategy='y', enabled=False),
        ]
        mgr = MultiExchangeManager(portfolio_configs=configs)
        assert 'okx' in mgr._states
        assert 'coinbase' not in mgr._states

    def test_record_trade_unknown_exchange(self, manager):
        # Should not crash
        manager.record_trade('binance', pnl=5.0, is_open=True)

    def test_open_positions_dont_go_negative(self, manager):
        manager.record_trade('okx', pnl=-1.0, is_open=False)
        assert manager._states['okx'].open_positions == 0


# ---------------------------------------------------------------------------
# Tests: Default configs
# ---------------------------------------------------------------------------

class TestDefaultConfigs:
    def test_default_configs_exist(self):
        assert len(DEFAULT_PORTFOLIO_CONFIGS) == 3

    def test_default_allocations_sum_to_100(self):
        total = sum(c.budget_allocation_pct for c in DEFAULT_PORTFOLIO_CONFIGS)
        assert total == 100.0

    def test_default_exchanges(self):
        exchanges = {c.exchange for c in DEFAULT_PORTFOLIO_CONFIGS}
        assert exchanges == {'hyperliquid', 'okx', 'coinbase'}
