"""Point-in-time features. Only uses data up to and including `date` (no lookahead)."""
import numpy as np
import pandas as pd


def rsi(series: pd.Series, n: int = 14) -> float:
    d = series.diff().dropna().tail(n)
    gain, loss = d.clip(lower=0).mean(), -d.clip(upper=0).mean()
    if loss == 0:
        return 100.0
    return float(100 - 100 / (1 + gain / loss))


def ticker_features(close: pd.Series) -> dict | None:
    s = close.dropna()
    if len(s) < 210:
        return None
    last = s.iloc[-1]
    rets = s.pct_change().dropna()
    return {
        "ret_1m": round(float(last / s.iloc[-21] - 1), 4),
        "ret_3m": round(float(last / s.iloc[-63] - 1), 4),
        "ret_6m": round(float(last / s.iloc[-126] - 1), 4),
        "vol_3m_ann": round(float(rets.tail(63).std() * np.sqrt(252)), 4),
        "vs_ma50": round(float(last / s.tail(50).mean() - 1), 4),
        "vs_ma200": round(float(last / s.tail(200).mean() - 1), 4),
        "rsi14": round(rsi(s), 1),
        "drawdown_from_1y_high": round(float(last / s.tail(252).max() - 1), 4),
    }


def snapshot(closes: pd.DataFrame, date) -> dict:
    hist = closes.loc[:date]
    feats = {}
    for t in hist.columns:
        f = ticker_features(hist[t])
        if f is not None:
            feats[t] = f
    return feats
