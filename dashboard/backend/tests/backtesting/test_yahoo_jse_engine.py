"""Yahoo JSE daily profile at the HourlyBacktester boundary (no network)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import engine as engine_module
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    MarketDataUnavailableError,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    JSE_TOP20_SYMBOLS,
    RULE_BASED_DECISION_SOURCE,
    YAHOO_JSE,
)

SAST = ZoneInfo("Africa/Johannesburg")
START = "2026-04-01"
END = "2026-04-14"
ZAR_PER_USD = 18.0


def make_daily_bars(symbols=JSE_TOP20_SYMBOLS, start=date(2026, 2, 20), end=date(2026, 4, 15)):
    stamps, current = [], start
    while current < end:
        if current.weekday() < 5:
            stamps.append(datetime.combine(current, time(17, 0), tzinfo=SAST))
        current += timedelta(days=1)
    index = pd.DatetimeIndex(stamps, name="timestamp")
    frames = {}
    for offset, symbol in enumerate(symbols):
        close = pd.Series([100.0 + offset + step for step in range(len(index))], index=index)
        frames[symbol] = pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}
        )
    return frames


class FakeYahoo:
    def __init__(self, bars):
        self.bars = bars
        self.calls, self.fx_calls = [], []

    def fetch_bars(self, symbols, start, end):
        self.calls.append((tuple(symbols), start, end))
        return {symbol: self.bars[symbol] for symbol in symbols if symbol in self.bars}

    def fetch_fx_rates(self, symbols, start, end):
        self.fx_calls.append((tuple(symbols), start, end))
        return {date(2026, 2, 20): ZAR_PER_USD}


class RecordingDB:
    def __init__(self):
        self.runs = []

    def insert_run(self, **kwargs):
        self.runs.append(kwargs)

    def insert_equity_points(self, run_id, points):
        pass

    def insert_trades(self, run_id, trades):
        pass


def _backtester(monkeypatch, provider, **kwargs):
    monkeypatch.setattr(
        engine_module, "create_market_data_provider", lambda _source, universe=None: provider
    )

    def fail_llm_client():
        raise AssertionError("a rule-based JSE run must not initialise an LLM")

    monkeypatch.setattr(engine_module, "make_llm_client", fail_llm_client)
    database = RecordingDB()
    monkeypatch.setattr(engine_module, "db", database)
    backtester = HourlyBacktester(
        START,
        END,
        session_id="yahoo-jse-engine-test",
        use_llm=False,
        data_source=YAHOO_JSE,
        decision_source=RULE_BASED_DECISION_SOURCE,
        **kwargs,
    )
    return backtester, database


def test_jse_rule_run_trades_daily_bars_in_usd(monkeypatch):
    provider = FakeYahoo(make_daily_bars())
    backtester, database = _backtester(monkeypatch, provider)

    assert tuple(backtester.symbols) == JSE_TOP20_SYMBOLS
    backtester.load_data()
    assert provider.fx_calls, "a ZAR run must load a USD/ZAR series"
    assert backtester.native_initial_capital == pytest.approx(
        backtester.initial_capital * ZAR_PER_USD
    )
    backtester.calculate_indicators()
    run_id, curve = backtester.run_agent_backtest()

    assert run_id and curve
    stamps = {pd.Timestamp(point["timestamp"]).tz_convert(SAST).time() for point in curve}
    assert stamps == {time(17, 0)}, "one mark per trading day, at the JSE close"
    metadata = database.runs[0]["metadata"]
    assert metadata["market"] == "ZA"
    assert metadata["fx_pair"] == "USD/ZAR"
    assert metadata["native_currency"] == "ZAR"


def test_jse_run_rejects_a_symbol_yahoo_returned_nothing_for(monkeypatch):
    bars = make_daily_bars()
    bars.pop("MTN.JO")
    backtester, _ = _backtester(monkeypatch, FakeYahoo(bars))
    with pytest.raises(MarketDataUnavailableError, match="MTN.JO"):
        backtester.load_data()


def test_jse_run_refuses_a_caller_chosen_stock_pool(monkeypatch):
    with pytest.raises(ValueError, match="stock_pool"):
        _backtester(monkeypatch, FakeYahoo({}), stock_pool="anything")


def test_cli_offers_the_yahoo_jse_source(tmp_path):
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "dashboard/scripts/backtest_hourly_agent.py", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )
    assert result.returncode == 0, result.stderr
    assert "yahoo_jse" in result.stdout


def test_jse_run_fetches_enough_daily_history_to_warm_sma50(monkeypatch):
    # sma50 needs 50 daily bars before the first decision; the hourly-sized
    # 30-calendar-day pad gave ~21, leaving sma50 and MACD unwarmed.
    provider = FakeYahoo(make_daily_bars())
    backtester, _ = _backtester(monkeypatch, provider)
    backtester.load_data()
    fetch_start = date.fromisoformat(provider.calls[0][1])
    assert (date.fromisoformat(START) - fetch_start).days >= 100
    assert date.fromisoformat(provider.fx_calls[0][1]) == fetch_start


def test_daily_pad_leaves_intraday_pads_unchanged():
    from dashboard.backend.infrastructure.market_data.provider import warmup_fetch_start

    assert warmup_fetch_start("2026-04-01") == "2026-03-02"
    assert warmup_fetch_start("2026-04-01", timeframe="60m") == "2026-03-02"
    assert warmup_fetch_start("2026-04-01", timeframe="1d") <= "2025-12-22"
