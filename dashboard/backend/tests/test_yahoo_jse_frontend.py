"""Contracts for the Yahoo JSE daily backtest UI: rule-based by default, AI model on request."""

from __future__ import annotations

import json
import shutil
import subprocess
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


_FAKE_SELECT = """
class FakeSelect {
  constructor(values, value) {
    this.options = values.map((v) => ({ value: v, textContent: v, remove() {} }));
    this.value = value;
    for (const option of this.options) this._own(option);
  }
  _own(option) {
    const select = this;
    option.remove = () => { select.options = select.options.filter((o) => o !== option); };
  }
  querySelector(selector) {
    const match = /value="([^"]+)"/.exec(selector);
    return this.options.find((o) => o.value === match[1]) || null;
  }
  insertBefore(option) { this._own(option); this.options.unshift(option); }
  get firstChild() { return this.options[0] || null; }
}
const document = {
  createElement: () => ({ value: '', textContent: '' }),
  elements: {},
  getElementById(id) { return this.elements[id] || null; },
};
const window = {};
"""

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _run_node(*lines: str):
    script = "\n".join(
        [
            js_const("YAHOO_JSE_SOURCE"),
            js_const("IFIND_ASHARE_SOURCE"),
            js_const("RULE_BASED_DECISION_SOURCE"),
            js_const("LLM_DECISION_SOURCE"),
            _FAKE_SELECT,
            *lines,
        ]
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
@pytest.mark.parametrize(
    "model, expected",
    [("rule_based", "rule_based"), ("anthropic/claude-sonnet-5.5", "llm"), ("", "rule_based")],
)
def test_jse_decision_source_follows_the_model_picker(model, expected):
    decided = _run_node(
        "function getIFindUniverseProfile() { return null; }",
        fn_body("function runBacktestModalDecisionSource"),
        "document.elements.marketDataSourceSelect = { value: YAHOO_JSE_SOURCE };",
        f"document.elements.modelSelect = {{ value: {json.dumps(model)} }};",
        "process.stdout.write(JSON.stringify(runBacktestModalDecisionSource()));",
    )
    assert decided == expected


@needs_node
def test_entering_jse_selects_rule_based_and_leaving_restores_the_model():
    states = _run_node(
        fn_body("function syncYahooJseModelChoice"),
        "const select = new FakeSelect(['model-a', 'model-b'], 'model-b');",
        "const seen = [];",
        "const snap = () => seen.push({ value: select.value, options: select.options.map((o) => o.value) });",
        "syncYahooJseModelChoice(true, select); snap();",
        "select.value = 'model-a'; syncYahooJseModelChoice(true, select); snap();",
        "syncYahooJseModelChoice(false, select); snap();",
        "process.stdout.write(JSON.stringify(seen));",
    )
    entered, reselected, left = states
    # Entering defaults to the free rule-based run; spending needs a deliberate pick.
    assert entered == {"value": "rule_based", "options": ["rule_based", "model-a", "model-b"]}
    # A model picked while on JSE stays picked; re-syncing does not reset it.
    assert reselected["value"] == "model-a"
    # Leaving removes the JSE-only option and restores what the US run had.
    assert left == {"value": "model-b", "options": ["model-a", "model-b"]}


def test_jse_model_picker_is_editable_and_rule_based_needs_no_key():
    mode = fn_body("function syncBacktestModelFieldMode")
    assert "isJse && modelSelect?.value === RULE_BASED_DECISION_SOURCE" in mode
    assert "!isSimulation && !isJse" not in mode
    submit = fn_body("function syncRunBacktestSubmitAvailability")
    assert "sourceHasRuleBasedChoice(dataSource)" in submit


def test_jse_keeps_its_rule_based_option_when_model_lists_reload():
    assert "YAHOO_JSE_SOURCE" in fn_body("function syncRunBacktestModelOptions")
    ui = fn_body("function syncMarketDataSourceUI")
    assert "syncYahooJseModelChoice(isJse" in ui
    assert "modelSelect.disabled = isSimulation;" in ui


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


def test_jse_source_label_no_longer_says_rule_based_only():
    picker = fn_body("async function loadMarketDataFeatures")
    assert "Yahoo JSE Top 20 (daily)" in picker
    assert "rule-based" not in picker


def test_jse_shows_its_own_universe_instead_of_the_us_picker(html):
    assert 'id="yahooJseUniverse"' in html
    panel = html[html.index('id="yahooJseUniverse"') - 80:html.index('id="yahooJseSymbolGrid"')]
    assert "hidden" in panel and "JSE Top 20 · 20 stocks" in panel
    sync = fn_body("function syncYahooJseUniverse")
    assert "YAHOO_JSE_SYMBOLS" in sync
    assert "builtinTab" in sync and "customTab" in sync
    assert "handleUniverseTabSwitch" in sync, "leaving JSE must restore the US tab the user had"
    assert "syncYahooJseUniverse(isJse" in fn_body("function syncMarketDataSourceUI")


def test_jse_period_hint_states_the_one_year_cap(html):
    assert 'id="runBacktestPeriodHint"' in html
    sync = fn_body("function syncYahooJseUniverse")
    assert "YAHOO_JSE_MAX_BACKTEST_DAYS" in sync
    assert "defaultText" in sync, "leaving JSE must restore the US hint"
