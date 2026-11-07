"""LLM decisions on the Yahoo JSE daily profile: what the model is told (no network, no spend)."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from dashboard.backend.domain.backtesting import engine as engine_module
from dashboard.backend.domain.backtesting.currency import CurrencyContext
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.infrastructure.llm.backtest_harness import (
    A_SHARE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    system_prompt_for_market,
)
from dashboard.backend.infrastructure.llm.validator import (
    create_custom_prompt,
    create_safe_prompt,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    IFIND_ASHARE,
    JSE_TOP20_SYMBOLS,
    LLM_DECISION_SOURCE,
    RULE_BASED_DECISION_SOURCE,
    YAHOO_JSE,
    get_market_profile,
    resolve_decision_source,
)
from dashboard.backend.tests.backtesting.test_yahoo_jse_engine import (
    END,
    START,
    FakeYahoo,
    RecordingDB,
    make_daily_bars,
)

ZA_SNAPSHOT = {"market": {"market": "ZA", "timeframe": "1d"}, "top_signals": []}
US_SNAPSHOT = {"market": {"market": "US", "timeframe": "60m"}, "top_signals": []}


# --- profile ---------------------------------------------------------------

def test_jse_profile_accepts_llm_but_defaults_to_rule_based():
    profile = get_market_profile(YAHOO_JSE)
    assert resolve_decision_source(profile) == RULE_BASED_DECISION_SOURCE
    assert resolve_decision_source(profile, "llm") == LLM_DECISION_SOURCE


# --- system prompt ---------------------------------------------------------

def test_jse_system_prompt_describes_the_jse_daily_paper_backtest():
    prompt = system_prompt_for_market({"market": "ZA"})
    assert "JSE" in prompt
    assert "daily" in prompt and "Africa/Johannesburg" in prompt
    assert "rand" in prompt.lower()
    assert "Securities Transfer Tax" in prompt
    assert "never sends real orders" in prompt
    assert "DJIA" not in prompt and "60m" not in prompt and "hour" not in prompt.lower()


def test_us_and_china_system_prompts_are_unchanged():
    assert system_prompt_for_market(None) == SYSTEM_PROMPT
    assert system_prompt_for_market({"market": "US"}) == SYSTEM_PROMPT
    assert system_prompt_for_market({"market": "CN"}) == A_SHARE_SYSTEM_PROMPT


# --- per-decision prompt ---------------------------------------------------

def test_jse_decision_prompts_name_the_jse_and_daily_data():
    safe = create_safe_prompt(ZA_SNAPSHOT, allowed_symbols=list(JSE_TOP20_SYMBOLS))
    custom = create_custom_prompt(ZA_SNAPSHOT, "Buy momentum.", allowed_symbols=list(JSE_TOP20_SYMBOLS))
    for prompt in (safe, custom):
        assert "DJIA" not in prompt
        assert "JSE" in prompt
        assert "hourly" not in prompt
    assert "historical daily data" in custom


def test_us_decision_prompts_are_byte_identical():
    # Leaderboard comparability: a US prompt must not change at all.
    assert "active DJIA portfolio trading agent" in create_safe_prompt(US_SNAPSHOT)
    assert "DJIA portfolio trading agent on historical hourly data" in create_custom_prompt(
        US_SNAPSHOT, "Buy momentum."
    )


# --- market context --------------------------------------------------------

def _context_for(data_source, fx_source, currency="ZAR"):
    backtester = object.__new__(HourlyBacktester)
    backtester.profile = get_market_profile(data_source)
    backtester.symbols = backtester.profile.symbols
    backtester.currency_context = CurrencyContext(
        native_currency=currency,
        reporting_currency="USD",
        timezone=backtester.profile.timezone,
        rates={date(2026, 4, 1): 18.0},
        fx_source=fx_source,
        fx_policy="daily_close_forward_fill",
    )
    return backtester._llm_market_context()


def test_jse_market_context_names_yahoo_as_the_fx_source():
    context = _context_for(YAHOO_JSE, "yahoo_usd_zar_daily_close")
    assert context["market"] == "ZA"
    assert context["timeframe"] == "1d"
    assert context["fx_pair"] == "USD/ZAR"
    assert "Yahoo" in context["fx_source"] and "iFinD" not in context["fx_source"]


def test_china_market_context_keeps_its_ifind_label():
    context = _context_for(IFIND_ASHARE, "ifind_history_currency_conversion", currency="CNY")
    assert context["fx_source"] == "iFinD Historical Conversion Rate"


# --- end to end ------------------------------------------------------------

class FakeLLMClient:
    """Anthropic-shaped client that records every request and always holds."""

    def __init__(self):
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        body = json.dumps({"actions": []})
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=body)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=100, output_tokens=10),
        )


def test_jse_llm_run_sends_the_jse_prompts_once_per_trading_day(monkeypatch):
    client = FakeLLMClient()
    monkeypatch.setattr(
        engine_module, "create_market_data_provider",
        lambda _source, universe=None: FakeYahoo(make_daily_bars()),
    )
    monkeypatch.setattr(engine_module, "make_llm_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(engine_module, "db", RecordingDB())
    backtester = HourlyBacktester(
        START, END, session_id="yahoo-jse-llm-test", use_llm=True,
        data_source=YAHOO_JSE, decision_source=LLM_DECISION_SOURCE,
    )

    backtester.load_data()
    backtester.calculate_indicators()
    backtester.run_agent_backtest()

    assert client.requests, "an LLM JSE run must call the model"
    trading_days = 10  # 1-14 April 2026 has 10 weekdays
    assert len(client.requests) <= trading_days
    for request in client.requests:
        assert request["system"] == system_prompt_for_market({"market": "ZA"})
        user = request["messages"][0]["content"]
        assert "DJIA" not in user and "hourly" not in user
