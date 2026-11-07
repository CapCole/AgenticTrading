"""JSE price data from Yahoo Finance with a local CSV cache."""
import hashlib
from pathlib import Path
import warnings
import pandas as pd

CACHE = Path(__file__).parent / "data_cache"


def _download(tickers, start, end):
    import yfinance as yf
    df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False, group_by="column")
    if df.empty:
        raise RuntimeError("No data returned. Check tickers and your internet connection.")
    out = {}
    for field in ("Open", "Close"):
        f = df[field]
        if isinstance(f, pd.Series):
            f = f.to_frame(tickers[0])
        out[field] = f
    return out["Open"], out["Close"]


GLITCH_RATIO = 0.02  # a print below 2% of its local median is a rands-vs-cents unit error


def _fix_unit_glitches(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo occasionally prints a .JO price in rands instead of cents (100x too small) for a
    day or two, e.g. SBK/STX40 in Apr 2025. Rescale those prints back to cents."""
    med = df.rolling(21, min_periods=5, center=True).median()
    glitch = (df / med) < GLITCH_RATIO
    if glitch.any().any():
        hits = {t: int(n) for t, n in glitch.sum().items() if n}
        warnings.warn(f"Rescaled rands-vs-cents price glitches (days per ticker): {hits}")
    return df.mask(glitch, df * 100)


def cache_key(tickers, start, end) -> str:
    """Short, stable filename stem for a price request (same in every run, unlike hash())."""
    raw = f"{'_'.join(sorted(tickers))}|{start}|{end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_prices(tickers, start, end=None, refresh=False):
    """Return (open, close) DataFrames in rands, indexed by date, one column per ticker.

    Yahoo quotes most .JO shares in cents (ZAc), so prices are divided by 100.
    Returns are unaffected either way.
    """
    CACHE.mkdir(exist_ok=True)
    key = cache_key(tickers, start, end)
    o_path, c_path = CACHE / f"{key}_open.csv", CACHE / f"{key}_close.csv"

    if o_path.exists() and c_path.exists() and not refresh:
        opens = pd.read_csv(o_path, index_col=0, parse_dates=True)
        closes = pd.read_csv(c_path, index_col=0, parse_dates=True)
    else:
        opens, closes = _download(list(tickers), start, end)
        opens.to_csv(o_path)
        closes.to_csv(c_path)

    missing = [t for t in tickers if t not in closes.columns or closes[t].dropna().empty]
    if missing:
        warnings.warn(f"No data for {missing}; skipping them.")
    keep = [t for t in tickers if t not in missing]
    opens, closes = opens[keep].copy(), closes[keep].copy()

    jse = [t for t in keep if t.endswith(".JO")]
    opens[jse] = _fix_unit_glitches(opens[jse])
    closes[jse] = _fix_unit_glitches(closes[jse])
    opens[jse] = opens[jse] / 100.0
    closes[jse] = closes[jse] / 100.0
    closes = closes.ffill()
    opens = opens.fillna(closes)
    return opens, closes
