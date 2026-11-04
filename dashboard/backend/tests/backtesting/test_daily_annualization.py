"""A daily-bar run annualises its Sharpe over 252 trading days, not hours."""

from __future__ import annotations

import pytest

from dashboard.backend.domain.backtesting.engine import HourlyBacktester


def _engine(source_timeframe: str, intraday_mode: bool) -> HourlyBacktester:
    engine = object.__new__(HourlyBacktester)
    engine.source_timeframe = source_timeframe
    engine.intraday_mode = intraday_mode
    return engine


def test_daily_source_annualises_over_trading_days():
    assert _engine("1d", intraday_mode=False)._annualization_periods() == 252


def test_hourly_source_keeps_the_historical_default():
    assert _engine("60m", intraday_mode=False)._annualization_periods() is None


def test_minute_source_uses_session_minutes():
    assert _engine("5m", intraday_mode=True)._annualization_periods() == pytest.approx(
        252 * 6.5 * 12
    )
