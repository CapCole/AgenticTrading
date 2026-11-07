"""Backtest copying US Congress members' disclosed stock buys, against holding SPY.

Usage:
  python run_congress.py                                  # all Congress, hold until they sell
  python run_congress.py --hold fixed --hold-days 60      # hold 60 days
  python run_congress.py --chamber senate                 # senators only
  python run_congress.py --no-costs                       # raw edge before trading costs

Research only. Prices from Yahoo Finance; delisted tickers are missing (survivorship bias).
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from agents import BuyAndHoldAgent
from backtest import metrics, run_backtest
from config import BacktestConfig, CostModel
from congress import CongressCopyAgent, build_lots, fetch_trades, parse_trades
from data import load_prices

BENCHMARK = "SPY"
CHUNK = 100
#: US shares bought from South Africa: commission 0.25% + VAT, no STT, 0.1% slippage. Estimates.
US_COSTS = CostModel(brokerage_pct=0.0025, vat_on_brokerage=0.15, stt_on_buys_pct=0.0,
                     other_levies_pct=0.0, slippage_pct=0.001)
NO_COSTS = CostModel(brokerage_pct=0.0, vat_on_brokerage=0.0, stt_on_buys_pct=0.0,
                     other_levies_pct=0.0, slippage_pct=0.0)


#: A chunk missing more than this share of its tickers was probably rate-limited, not delisted.
MAX_MISSING = 0.35
RETRIES = 3
PAUSE_S = 20


def _load_chunk(chunk, start, end):
    """One chunk, re-downloaded with a pause when Yahoo rate-limits it (failures get cached)."""
    for attempt in range(RETRIES + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, c = load_prices(chunk, start, end, refresh=attempt > 0)
        missing = 1 - c.shape[1] / len(chunk)
        if missing <= MAX_MISSING or attempt == RETRIES:
            return o, c, missing
        print(f"  {missing:.0%} of a chunk missing (rate limit?); retrying in {PAUSE_S * (attempt + 1)}s")
        time.sleep(PAUSE_S * (attempt + 1))


def load_us_prices(tickers, start, end):
    """Opens and closes for many US tickers, downloaded and cached in chunks."""
    opens, closes = [], []
    tickers = sorted(tickers)
    for i in range(0, len(tickers), CHUNK):
        o, c, missing = _load_chunk(tickers[i:i + CHUNK], start, end)
        if missing > MAX_MISSING:
            print(f"  warning: chunk {i // CHUNK + 1} still {missing:.0%} missing after retries")
        opens.append(o)
        closes.append(c)
    return pd.concat(opens, axis=1), pd.concat(closes, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2026-10-31")
    p.add_argument("--hold", choices=["fixed", "until_sale"], default="until_sale")
    p.add_argument("--hold-days", type=int, default=365)
    p.add_argument("--chamber", choices=["house", "senate"], default=None)
    p.add_argument("--rebalance", type=int, default=5, help="trading days between rebalances (5 = weekly)")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--max-position", type=float, default=0.10)
    p.add_argument("--top-n", type=int, default=20, help="hold the N most-bought tickers (0 = all)")
    p.add_argument("--no-costs", action="store_true")
    a = p.parse_args()

    rows = fetch_trades()
    trades = parse_trades(rows, chamber=a.chamber)
    trades = trades[trades["filing_date"] <= a.end]
    lots = build_lots(trades, hold=a.hold, hold_days=a.hold_days)
    lots = lots[lots["end"] > pd.Timestamp(a.start)]
    # One price universe for every variant (all Congress buys), so the cache is reused.
    everyone = parse_trades(rows)
    universe = everyone[(everyone["side"] == "buy") & (everyone["filing_date"] <= a.end)
                        & (everyone["filing_date"] >= pd.Timestamp(a.start) - pd.Timedelta(days=365))]
    opens, closes = load_us_prices(set(universe["ticker"]) | set(lots["ticker"]), a.start, a.end)
    bo, bc = load_prices([BENCHMARK], a.start, a.end)

    costs = NO_COSTS if a.no_costs else US_COSTS
    cfg = BacktestConfig(start=a.start, end=a.end, capital=a.capital, rebalance_every=a.rebalance,
                         max_position=a.max_position, risk_free_rate=0.04, costs=costs)
    curve, tr, total_costs = run_backtest(CongressCopyAgent(lots, top_n=a.top_n or None), opens, closes, cfg, warmup=0)
    bench, _, _ = run_backtest(BuyAndHoldAgent(BENCHMARK), bo, bc, cfg, warmup=0)

    label = (f"{a.chamber or 'congress'}_{a.hold}{a.hold_days}_top{a.top_n or 'all'}"
             f"{'_nocost' if a.no_costs else ''}")
    results = {
        "variant": label,
        "tickers_with_prices": int(closes.shape[1]),
        "tickers_bought": int(lots["ticker"].nunique()),
        "price_coverage": round(len(set(lots["ticker"]) & set(closes.columns))
                                / max(1, lots["ticker"].nunique()), 3),
        "copy_congress": {**metrics(curve, cfg.risk_free_rate), "total_costs": round(total_costs, 2),
                          "num_trades": len(tr)},
        "benchmark_SPY": metrics(bench, cfg.risk_free_rate),
    }
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"congress_{label}.json").write_text(json.dumps(results, indent=2))
    curve.to_csv(out / f"congress_{label}_equity.csv")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
