"""Source contracts for the Yahoo JSE daily backtest UI (rule-based only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.backend.api.routers import backtests as bt
from dashboard.backend.tests._frontend_source import fn_body, js_const

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


@pytest.fixture(scope="module")
def html() -> str:
    return (_FRONTEND / "app.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return (_FRONTEND / "app.js").read_text(encoding="utf-8")


def test_jse_option_is_feature_gated_in_javascript_not_html(html, js):
    assert "yahoo_jse" not in html
    assert "features.yahoo_jse_enabled === true" in js
    picker = fn_body("async function loadMarketDataFeatures")
    assert "window.YAHOO_JSE_ENABLED" in picker
    assert "YAHOO_JSE_SOURCE" in picker


def test_jse_constants_mirror_the_server():
    assert js_const("YAHOO_JSE_SOURCE") == "const YAHOO_JSE_SOURCE = 'yahoo_jse';"
    assert js_const("YAHOO_JSE_TIMEFRAME") == "const YAHOO_JSE_TIMEFRAME = '1d';"
    assert js_const("YAHOO_JSE_UNIVERSE") == "const YAHOO_JSE_UNIVERSE = 'jse_top20';"
    assert js_const("YAHOO_JSE_MAX_BACKTEST_DAYS") == (
        f"const YAHOO_JSE_MAX_BACKTEST_DAYS = {bt.max_backtest_days(bt.YAHOO_JSE)};"
    )


def test_jse_runs_are_rule_based_in_the_modal():
    decision = fn_body("function runBacktestModalDecisionSource")
    assert "dataSource === YAHOO_JSE_SOURCE" in decision
    mode = fn_body("function syncBacktestModelFieldMode")
    assert "isJse" in mode


def test_jse_request_sends_profile_universe_and_daily_timeframe():
    run = fn_body("async function runBacktest")
    assert "params.set('universe', YAHOO_JSE_UNIVERSE)" in run
    assert "params.set('timeframe', YAHOO_JSE_TIMEFRAME)" in run
    assert "isJse ? YAHOO_JSE_MAX_BACKTEST_DAYS : MAX_BACKTEST_DAYS" in run


def test_jse_never_sends_a_stock_pool_or_us_assets():
    assert "YAHOO_JSE_SOURCE" in fn_body("function getSelectedStockPoolRequest")
    assert "YAHOO_JSE_SYMBOLS" in fn_body("function getSelectedAssets")


def test_jse_runs_get_their_own_badge():
    assert "Yahoo JSE · daily" in fn_body("function renderBacktestDataSourceBadge")


def test_jse_mode_applies_a_year_and_restores_previous_dates():
    sync = fn_body("function syncYahooJseDates")
    assert "YAHOO_JSE_START_DATE" in sync and "YAHOO_JSE_END_DATE" in sync
    assert "YAHOO_JSE_PREVIOUS_DATES = null" in sync
    ui = fn_body("function syncMarketDataSourceUI")
    assert "syncYahooJseDates(isJse" in ui
    assert "universeTabs.hidden = isIFind || isJse" in ui
