"""JSE daily bars from Yahoo Finance (``.JO`` tickers), for research backtests.

Data only: there is no JSE broker behind this source. Yahoo quotes JSE shares
in cents (ZAc); bars are returned in rands, one per trading day, stamped at the
17:00 SAST close (close-stamped, so ``sessions.is_in_session`` keeps them).

Yahoo occasionally prints a day in rands instead of cents -- 100x too small
(SBK and the STX40 ETF on 2025-04-25, VOD on 2025-01-10). Such a print is
rescaled and its date listed in ``frame.attrs["unit_rescaled_dates"]``: a
labelled correction, never a silent one.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Any, Callable, Sequence

import pandas as pd
import pytz

from .profiles import MarketProfile

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_YAHOO_FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}

#: Yahoo's ZAR-per-USD quote.
USD_ZAR_TICKER = "ZAR=X"
#: JSE closing auction ends at 17:00 SAST; daily bars are stamped there.
JSE_CLOSE = time(17, 0)
#: A print below this fraction of its local median is a rands-vs-cents error.
UNIT_GLITCH_RATIO = 0.02
#: Bars either side of a print used for the local median.
_GLITCH_WINDOW = 21
_CENTS_PER_RAND = 100.0

DateInput = str | date | datetime
Downloader = Callable[..., pd.DataFrame]


class YahooJseDataError(ValueError):
    """Raised when Yahoo returns nothing usable or a request is malformed."""


def _default_downloader(tickers: Sequence[str], **kwargs: Any) -> pd.DataFrame:
    import yfinance as yf  # optional at import time; pinned in requirements.in

    return yf.download(list(tickers), **kwargs)


def _iso(value: DateInput) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise YahooJseDataError(f"expected a YYYY-MM-DD date, got {value!r}") from exc


def _field_frame(raw: pd.DataFrame, field: str, tickers: Sequence[str]) -> pd.DataFrame:
    """One Yahoo field as a ``date x ticker`` frame, single- or multi-ticker."""
    if isinstance(raw.columns, pd.MultiIndex):
        return raw[field]
    return raw[[field]].rename(columns={field: tickers[0]})


def rescale_unit_glitches(prices: pd.Series) -> tuple[pd.Series, list[str]]:
    """Return ``prices`` with rands-for-cents prints scaled back, and their dates."""
    median = prices.rolling(_GLITCH_WINDOW, min_periods=3, center=True).median()
    glitch = (prices / median) < UNIT_GLITCH_RATIO
    dates = [stamp.date().isoformat() for stamp in prices.index[glitch.fillna(False)]]
    return prices.mask(glitch, prices * _CENTS_PER_RAND), dates


class YahooJseProvider:
    """``MarketDataProvider`` for the ``yahoo_jse`` profile (daily, ZAR)."""

    source_timeframe = "1d"

    def __init__(self, *, profile: MarketProfile, downloader: Downloader | None = None):
        self.profile = profile
        self._download = downloader or _default_downloader
        self._timezone = pytz.timezone(profile.timezone)

    def fetch_bars(
        self, symbols: Sequence[str], start: DateInput, end: DateInput
    ) -> dict[str, pd.DataFrame]:
        """Daily OHLCV in rands over the half-open ``[start, end)``.

        A symbol Yahoo has no rows for is left out; the engine reports it.
        """
        tickers = self._validate_universe(symbols)
        raw = self._download(
            tickers,
            start=_iso(start),
            end=_iso(end),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw is None or raw.empty:
            raise YahooJseDataError(f"Yahoo returned no data for {list(tickers)!r}")
        return {
            ticker: frame
            for ticker in tickers
            if (frame := self._symbol_frame(raw, ticker, tickers)) is not None
        }

    def fetch_fx_rates(
        self, symbols: Sequence[str], start: DateInput, end: DateInput
    ) -> dict[date, float]:
        """Rand-per-dollar closes by date over ``[start, end)``."""
        self._validate_universe(symbols)
        raw = self._download(
            [USD_ZAR_TICKER],
            start=_iso(start),
            end=_iso(end),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw is None or raw.empty:
            raise YahooJseDataError("Yahoo returned no USD/ZAR rates")
        closes = _field_frame(raw, "Close", [USD_ZAR_TICKER]).iloc[:, 0].dropna()
        return {stamp.date(): float(rate) for stamp, rate in closes.items() if rate > 0}

    def _symbol_frame(
        self, raw: pd.DataFrame, ticker: str, tickers: Sequence[str]
    ) -> pd.DataFrame | None:
        columns = {}
        for yahoo_field, name in _YAHOO_FIELDS.items():
            field = _field_frame(raw, yahoo_field, tickers)
            if ticker not in field.columns:
                return None
            columns[name] = field[ticker]
        frame = pd.DataFrame(columns).dropna(subset=list(_PRICE_COLUMNS), how="all")
        if frame.empty:
            return None

        rescaled_dates: set[str] = set()
        for column in _PRICE_COLUMNS:
            fixed, dates = rescale_unit_glitches(frame[column])
            frame[column] = fixed / _CENTS_PER_RAND
            rescaled_dates.update(dates)
        frame["volume"] = frame["volume"].fillna(0.0)

        frame.index = pd.DatetimeIndex(
            [
                self._timezone.localize(datetime.combine(stamp.date(), JSE_CLOSE))
                for stamp in frame.index
            ],
            name="timestamp",
        )
        frame = frame.loc[:, list(OHLCV_COLUMNS)]
        if rescaled_dates:
            logger.warning(
                "yahoo_jse: rescaled rands-for-cents prints for %s on %s",
                ticker,
                sorted(rescaled_dates),
            )
        frame.attrs["unit_rescaled_dates"] = sorted(rescaled_dates)
        return frame

    def _validate_universe(self, symbols: Sequence[str]) -> tuple[str, ...]:
        tickers = tuple(str(symbol).strip().upper() for symbol in symbols)
        outside = [ticker for ticker in tickers if ticker not in self.profile.symbols]
        if not tickers or outside:
            raise YahooJseDataError(
                f"symbols outside the {self.profile.universe} universe: {outside!r}"
            )
        return tickers
