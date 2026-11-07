"""Copy-Congress strategy: buy what US Congress members disclose buying, once it is public.

Data: the Kadoa congress-trading-monitor dataset (MIT), parsed from the official House Clerk
and Senate eFD STOCK Act filings. `fetch_trades()` downloads it into data_cache/congress/.

No look-ahead: a purchase is usable only from its filing date (when it became public), so the
backtester, which decides at the close and trades at the next open, enters the day after filing.
"""
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).parent / "data_cache" / "congress"
_REPO = "kadoa-org/congress-trading-monitor"
_RAW = f"https://raw.githubusercontent.com/{_REPO}/main/public/data/filer/"
_LISTING = f"https://api.github.com/repos/{_REPO}/contents/public/data/filer"
_TICKER = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")
BUY_TYPES = {"Purchase"}
SELL_TYPES = {"Sale (Full)", "Sale (Partial)", "Sale"}


def normalize_ticker(raw):
    """Yahoo-style ticker (BRK.B -> BRK-B), or None for anything that isn't a plain US ticker."""
    t = str(raw or "").strip().upper().replace(".", "-")
    return t if _TICKER.match(t) else None


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "jse-lab-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_trades(refresh=False):
    """Download every filer file once; return the raw trade rows with chamber and branch attached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if refresh or not any(CACHE.glob("*.json")):
        names = [f["name"] for f in json.loads(_get(_LISTING)) if f["name"].endswith(".json")]
        def save(name):
            (CACHE / name).write_bytes(_get(_RAW + name))
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(save, names))
    rows = []
    for path in sorted(CACHE.glob("*.json")):
        data = json.loads(path.read_text())
        filer = data.get("filer") or {}
        for t in data.get("trades", []):
            rows.append({**t, "_chamber": filer.get("chamber"), "_branch": filer.get("branch")})
    return rows


def parse_trades(rows, chamber=None):
    """Congress stock buys and sales as a DataFrame: filer_id, ticker, side, filing_date."""
    out = []
    for r in rows:
        if r.get("_branch") != "congress" or (chamber and r.get("_chamber") != chamber):
            continue
        kind = r.get("transaction_type")
        side = "buy" if kind in BUY_TYPES else "sell" if kind in SELL_TYPES else None
        ticker = normalize_ticker(r.get("ticker"))
        filed = pd.to_datetime(r.get("filing_date"), errors="coerce")
        if side and ticker and not pd.isna(filed):
            out.append({"filer_id": r["filer_id"], "ticker": ticker, "side": side, "filing_date": filed})
    cols = ["filer_id", "ticker", "side", "filing_date"]
    return pd.DataFrame(out, columns=cols).sort_values("filing_date", kind="stable").reset_index(drop=True)


def build_lots(trades, hold="until_sale", hold_days=365):
    """One lot per disclosed buy: [start, end) where start is the filing date.

    hold="fixed": end = start + hold_days.
    hold="until_sale": end = the same member's next filed sale of that ticker, capped at hold_days.
    """
    if hold not in ("fixed", "until_sale"):
        raise ValueError(f"hold must be 'fixed' or 'until_sale', got {hold!r}")
    buys = trades[trades["side"] == "buy"]
    sells = trades[trades["side"] == "sell"]
    sale_dates = sells.groupby(["filer_id", "ticker"])["filing_date"].apply(sorted).to_dict()
    lots = []
    for b in buys.itertuples():
        end = b.filing_date + pd.Timedelta(days=hold_days)
        if hold == "until_sale":
            later = [d for d in sale_dates.get((b.filer_id, b.ticker), []) if d > b.filing_date]
            if later:
                end = min(end, later[0])
        lots.append({"ticker": b.ticker, "start": b.filing_date, "end": end})
    return pd.DataFrame(lots, columns=["ticker", "start", "end"])


class CongressCopyAgent:
    """Equal-weight the tickers with open lots (filed buy, not yet ended) that have a price.

    top_n keeps the tickers with the most open buys (most members currently in), ties broken by
    the most recent filing. Without it, ~500 names are open on a typical day: positions too small
    to be worth their trading costs.
    """
    name = "congress"

    def __init__(self, lots, top_n=None):
        self.lots = lots
        self.top_n = top_n

    def decide(self, date, closes, holdings, max_position):
        date = pd.Timestamp(date)
        open_lots = self.lots[(self.lots["start"] <= date) & (self.lots["end"] > date)]
        priced = closes.loc[:date].iloc[-1] if len(closes.loc[:date]) else pd.Series(dtype=float)
        has_price = [t for t in set(open_lots["ticker"]) if t in closes.columns and pd.notna(priced.get(t))]
        open_lots = open_lots[open_lots["ticker"].isin(has_price)]
        if self.top_n:
            ranked = (open_lots.groupby("ticker")["start"].agg(["count", "max"])
                      .sort_values(["count", "max"], ascending=False))
            tickers = sorted(ranked.index[: self.top_n])
        else:
            tickers = sorted(set(open_lots["ticker"]))
        if not tickers:
            return {}
        weight = min(1.0 / len(tickers), max_position)
        return {t: weight for t in tickers}
