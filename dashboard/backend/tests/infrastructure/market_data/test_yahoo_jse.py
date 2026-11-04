"""Yahoo JSE daily provider: cents to rands, close stamps, glitch guard, FX.

No network: every test injects a fake ``yfinance.download``.
"""

from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from dashboard.backend.infrastructure.market_data.profiles import (
    YAHOO_JSE,
    get_market_profile,
)
from dashboard.backend.infrastructure.market_data.yahoo_jse import (
    YahooJseDataError,
    YahooJseProvider,
)

_DAYS = pd.to_datetime(["2025-04-22", "2025-04-23", "2025-04-24", "2025-04-25", "2025-04-28"])


def _download_result(prices_by_symbol: dict[str, list[float]]) -> pd.DataFrame:
    """A ``group_by="column"`` frame: (field, symbol) MultiIndex columns."""
    columns = {}
    for symbol, closes in prices_by_symbol.items():
        for field in ("Open", "High", "Low", "Close"):
            columns[(field, symbol)] = closes
        columns[("Volume", symbol)] = [1000.0] * len(closes)
    frame = pd.DataFrame(columns, index=_DAYS[: len(next(iter(prices_by_symbol.values())))])
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def _provider(result: pd.DataFrame, calls: list | None = None) -> YahooJseProvider:
    def fake_download(tickers, **kwargs):
        if calls is not None:
            calls.append((tuple(tickers), kwargs))
        return result

    return YahooJseProvider(profile=get_market_profile(YAHOO_JSE), downloader=fake_download)


def test_fetch_bars_converts_cents_to_rands_and_stamps_the_jse_close():
    provider = _provider(_download_result({"NPN.JO": [500000.0, 510000.0, 505000.0]}))

    bars = provider.fetch_bars(["NPN.JO"], "2025-04-22", "2025-04-25")

    frame = bars["NPN.JO"]
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame["close"].tolist() == [5000.0, 5100.0, 5050.0]
    first = frame.index[0]
    assert str(first.tz) == "Africa/Johannesburg"
    assert first.time() == time(17, 0)
    assert first.date() == date(2025, 4, 22)


def test_fetch_bars_requests_adjusted_daily_bars_for_the_half_open_window():
    calls: list = []
    provider = _provider(_download_result({"NPN.JO": [500000.0]}), calls)

    provider.fetch_bars(["NPN.JO"], "2025-04-22", "2025-04-23")

    (tickers, kwargs), = calls
    assert tickers == ("NPN.JO",)
    assert kwargs["interval"] == "1d"
    assert kwargs["auto_adjust"] is True
    assert (kwargs["start"], kwargs["end"]) == ("2025-04-22", "2025-04-23")


def test_rands_for_cents_glitch_is_rescaled_and_labelled():
    # SBK on 2025-04-25: Yahoo printed rands, 100x too small.
    provider = _provider(
        _download_result({"SBK.JO": [23000.0, 23100.0, 23050.0, 231.0, 23200.0]})
    )

    frame = provider.fetch_bars(["SBK.JO"], "2025-04-22", "2025-04-29")["SBK.JO"]

    assert frame["close"].tolist() == [230.0, 231.0, 230.5, 231.0, 232.0]
    assert frame.attrs["unit_rescaled_dates"] == ["2025-04-25"]


def test_symbol_outside_the_profile_universe_is_refused():
    provider = _provider(_download_result({"NPN.JO": [1.0]}))
    with pytest.raises(YahooJseDataError, match="universe"):
        provider.fetch_bars(["AAPL"], "2025-04-22", "2025-04-23")


def test_symbol_with_no_rows_is_left_out_for_the_engine_to_report():
    result = _download_result({"NPN.JO": [500000.0, 510000.0], "MTN.JO": [float("nan")] * 2})
    bars = _provider(result).fetch_bars(["NPN.JO", "MTN.JO"], "2025-04-22", "2025-04-24")
    assert set(bars) == {"NPN.JO"}


def test_empty_download_raises():
    with pytest.raises(YahooJseDataError, match="no data"):
        _provider(pd.DataFrame()).fetch_bars(["NPN.JO"], "2025-04-22", "2025-04-23")


def test_fetch_fx_rates_returns_rand_per_dollar_by_date():
    calls: list = []
    fx = pd.DataFrame({"Close": [18.5, 18.7]}, index=_DAYS[:2])
    provider = _provider(fx, calls)

    rates = provider.fetch_fx_rates(["NPN.JO"], "2025-04-22", "2025-04-24")

    assert rates == {date(2025, 4, 22): 18.5, date(2025, 4, 23): 18.7}
    assert calls[0][0] == ("ZAR=X",)
