"""Tests for trailing stop, breakeven move, and partial profit taking."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from nct.config import RiskConfig
from nct.exchange.models import OrderResponse, OrderStatus, Side
from nct.executor import OrderExecutor
from nct.portfolio.tracker import TrackedTrade


def _make_trade(**overrides) -> TrackedTrade:
    defaults = {
        'trade_id': 1,
        'inst_id': 'BTC-USD',
        'side': 'buy',
        'size': Decimal('0.01'),
        'entry_price': Decimal('65000'),
        'fee': Decimal('2.6'),
        'stop_loss_algo_id': 'sl_abc',
        'take_profit_algo_id': 'tp_xyz',
        'stop_loss_price': Decimal('63000'),
        'take_profit_price': Decimal('68000'),
        'high_water_mark': None,
        'breakeven_applied': False,
        'partial_stages_fired': None,
    }
    defaults.update(overrides)
    return TrackedTrade(**defaults)


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.cancel_order = AsyncMock()
    client.place_stop_loss = AsyncMock(return_value='new_sl')
    client.place_take_profit = AsyncMock(return_value='new_tp')
    client.place_order = AsyncMock(return_value=MagicMock())
    return client


@pytest.fixture
def mock_portfolio():
    pt = MagicMock()
    pt.open_trades = {}
    return pt


@pytest.fixture
def executor(mock_client, mock_portfolio):
    return OrderExecutor(
        client=mock_client,
        portfolio=mock_portfolio,
        budget_manager=MagicMock(),
        protection_manager=MagicMock(),
    )


# -- Trailing Stop (#65) ------------------------------------------------------

class TestTrailingStop:
    @pytest.mark.asyncio
    async def test_trailing_activates_after_threshold(self, executor, mock_portfolio, mock_client):
        trade = _make_trade(high_water_mark=Decimal('67000'))
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(
            trailing_stop=True,
            trailing_stop_activation_pct=Decimal('2.0'),
            trailing_stop_delta_pct=Decimal('1.0'),
        )

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('67000')},
            risk_config=config,
        )

        # HWM=67000, activation=65000*1.02=66300 → activated
        # trailing SL = 67000 * 0.99 = 66330 > current SL 63000 → should update
        mock_client.place_stop_loss.assert_called_once()
        assert trade.stop_loss_price == Decimal('67000') * (1 - Decimal('1.0') / 100)

    @pytest.mark.asyncio
    async def test_trailing_does_not_activate_below_threshold(self, executor, mock_portfolio, mock_client):
        trade = _make_trade(high_water_mark=Decimal('65500'))
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(
            trailing_stop=True,
            trailing_stop_activation_pct=Decimal('2.0'),
            trailing_stop_delta_pct=Decimal('1.0'),
        )

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('65500')},
            risk_config=config,
        )

        # HWM=65500, activation=66300 → NOT activated
        mock_client.place_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_trailing_never_moves_sl_down(self, executor, mock_portfolio, mock_client):
        trade = _make_trade(
            stop_loss_price=Decimal('66500'),
            high_water_mark=Decimal('67000'),
        )
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(
            trailing_stop=True,
            trailing_stop_activation_pct=Decimal('2.0'),
            trailing_stop_delta_pct=Decimal('1.0'),
        )

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('66800')},
            risk_config=config,
        )

        # trailing SL = 67000*0.99 = 66330 < current SL 66500 → should NOT update
        mock_client.place_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_trailing_disabled_no_action(self, executor, mock_portfolio, mock_client):
        trade = _make_trade(high_water_mark=Decimal('70000'))
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(trailing_stop=False)

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('70000')},
            risk_config=config,
        )

        mock_client.place_stop_loss.assert_not_called()


# -- Breakeven Move (#66) -----------------------------------------------------

class TestBreakevenMove:
    @pytest.mark.asyncio
    async def test_breakeven_triggers_at_threshold(self, executor, mock_portfolio, mock_client):
        trade = _make_trade()
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(breakeven_trigger_pct=Decimal('1.5'))

        # Price at +2% > +1.5% trigger
        price = Decimal('66300')
        await executor.check_exit_management(
            current_prices={'BTC-USD': price},
            risk_config=config,
        )

        assert trade.breakeven_applied is True
        mock_client.cancel_order.assert_called()
        mock_client.place_stop_loss.assert_called_once()

    @pytest.mark.asyncio
    async def test_breakeven_does_not_trigger_below(self, executor, mock_portfolio, mock_client):
        trade = _make_trade()
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(breakeven_trigger_pct=Decimal('2.0'))

        # Price at +0.5% < +2% trigger
        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('65325')},
            risk_config=config,
        )

        assert trade.breakeven_applied is False
        mock_client.place_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_breakeven_fires_only_once(self, executor, mock_portfolio, mock_client):
        trade = _make_trade(breakeven_applied=True)
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(breakeven_trigger_pct=Decimal('1.0'))

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('70000')},
            risk_config=config,
        )

        mock_client.place_stop_loss.assert_not_called()

    @pytest.mark.asyncio
    async def test_breakeven_disabled_when_zero(self, executor, mock_portfolio, mock_client):
        trade = _make_trade()
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(breakeven_trigger_pct=Decimal('0'))

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('70000')},
            risk_config=config,
        )

        assert trade.breakeven_applied is False


# -- Partial Profit Taking (#67) -----------------------------------------------

class TestPartialProfitTaking:
    @pytest.mark.asyncio
    async def test_partial_fires_at_trigger(self, executor, mock_portfolio, mock_client):
        trade = _make_trade(size=Decimal('0.10'))
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(
            partial_tp=[{'pct': 3.0, 'close_fraction': 0.5}],
        )

        # Price at +4% > +3% trigger
        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('67600')},
            risk_config=config,
        )

        # Should have placed a market sell for 50% of position
        mock_client.place_order.assert_called_once()
        assert trade.size == Decimal('0.05')  # 50% remaining
        assert 0 in trade.partial_stages_fired

    @pytest.mark.asyncio
    async def test_partial_does_not_fire_below_trigger(self, executor, mock_portfolio, mock_client):
        trade = _make_trade(size=Decimal('0.10'))
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(
            partial_tp=[{'pct': 5.0, 'close_fraction': 0.5}],
        )

        # Price at +2% < +5% trigger
        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('66300')},
            risk_config=config,
        )

        mock_client.place_order.assert_not_called()
        assert trade.size == Decimal('0.10')

    @pytest.mark.asyncio
    async def test_partial_fires_only_once_per_stage(self, executor, mock_portfolio, mock_client):
        trade = _make_trade(size=Decimal('0.05'), partial_stages_fired=[0])
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(
            partial_tp=[{'pct': 3.0, 'close_fraction': 0.5}],
        )

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('70000')},
            risk_config=config,
        )

        mock_client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_partial_tp_no_action(self, executor, mock_portfolio, mock_client):
        trade = _make_trade()
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig(partial_tp=[])

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('70000')},
            risk_config=config,
        )

        mock_client.place_order.assert_not_called()


# -- High Water Mark tracking --------------------------------------------------

class TestHighWaterMark:
    @pytest.mark.asyncio
    async def test_hwm_updates_on_new_high(self, executor, mock_portfolio):
        trade = _make_trade(high_water_mark=Decimal('66000'))
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig()

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('67000')},
            risk_config=config,
        )

        assert trade.high_water_mark == Decimal('67000')

    @pytest.mark.asyncio
    async def test_hwm_does_not_decrease(self, executor, mock_portfolio):
        trade = _make_trade(high_water_mark=Decimal('68000'))
        mock_portfolio.open_trades = {'BTC-USD': trade}

        config = RiskConfig()

        await executor.check_exit_management(
            current_prices={'BTC-USD': Decimal('66000')},
            risk_config=config,
        )

        assert trade.high_water_mark == Decimal('68000')
