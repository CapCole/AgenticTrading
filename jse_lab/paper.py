"""Forward paper trading: virtual portfolios run weekly on live prices, no real orders.

Same rules as backtest.py: decide at a completed close, fill at the next trading day's
open with EasyEquities costs. Each run (1) fills last run's orders, (2) marks to market,
(3) decides new targets. Ledgers live in paper/<name>.json; commit them each week so the
record cannot be rewritten after the fact.

Usage: python paper.py            (run weekly, any time after the JSE close)
Not financial advice. Nothing here sends orders to a broker.
"""
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agents import BuyAndHoldAgent, MomentumAgent
from config import BENCHMARK, DEFAULT_UNIVERSE, BacktestConfig

PAPER_DIR = Path(__file__).parent / "paper"
JSE_TZ = "Africa/Johannesburg"
CLOSE_SETTLED = (17, 30)  # JSE closes 17:00; Yahoo's close is reliable after this
HISTORY_START = "2023-01-01"  # enough for 200-day averages and 6-month returns


class HoldAgent:
    """Equal-weight the universe once, then never trade again (buy-and-hold baseline)."""
    name = "equal_weight_hold"

    def __init__(self, tickers):
        self.tickers = list(tickers)

    def decide(self, date, closes, holdings, max_position):
        if holdings:
            return dict(holdings)
        return {t: 1.0 / len(self.tickers) for t in self.tickers}


def new_ledger(name: str, cfg: BacktestConfig) -> dict:
    return {"name": name, "start": None, "capital": cfg.capital, "cash": cfg.capital,
            "shares": {}, "pending": None, "history": [], "trades": []}


def completed_bars(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop today's row until the JSE close has settled, so a partial day is never used."""
    local = now.tz_convert(JSE_TZ)
    settled = (local.hour, local.minute) >= CLOSE_SETTLED
    today = pd.Timestamp(local.date())
    return df[df.index <= today] if settled else df[df.index < today]


def _equity(cash: float, shares: dict, px: pd.Series) -> float:
    return cash + sum(q * px[t] for t, q in shares.items() if not np.isnan(px[t]))


def _fill(led: dict, day: pd.Timestamp, px: pd.Series, cfg: BacktestConfig) -> None:
    """Trade the pending target weights at this day's open prices (sells first)."""
    weights = led["pending"]["weights"]
    equity = _equity(led["cash"], led["shares"], px)
    orders = []
    for t in sorted(set(led["shares"]) | set(weights)):
        if t not in px or np.isnan(px[t]):
            continue
        diff = weights.get(t, 0.0) * equity - led["shares"].get(t, 0.0) * px[t]
        if abs(diff) >= cfg.min_trade_value:
            orders.append((t, diff))
    for t, diff in sorted(orders, key=lambda o: o[1]):
        is_buy = diff > 0
        cost = cfg.costs.cost(diff, is_buy)
        if is_buy:
            diff = min(diff, led["cash"] - cost)
            if diff <= 0:
                continue
        qty = led["shares"].get(t, 0.0) + diff / px[t]
        if qty > 1e-9:
            led["shares"][t] = qty
        else:
            led["shares"].pop(t, None)
        led["cash"] -= diff + cost
        led["trades"].append({"date": str(day.date()), "ticker": t,
                              "value": round(diff, 2), "cost": round(cost, 2)})
    led["pending"] = None


def step(ledger: dict, opens: pd.DataFrame, closes: pd.DataFrame, agent,
         cfg: BacktestConfig) -> dict:
    """Return an updated copy of the ledger; the input is left unchanged."""
    led = copy.deepcopy(ledger)
    last = closes.index[-1]
    if led["history"] and led["history"][-1]["date"] >= str(last.date()):
        return led  # nothing new since the last run

    if led["pending"]:
        decided = pd.Timestamp(led["pending"]["decided"])
        later = opens.index[opens.index > decided]
        if len(later):
            _fill(led, later[0], opens.loc[later[0]], cfg)

    px = closes.loc[last]
    equity = _equity(led["cash"], led["shares"], px)
    led["start"] = led["start"] or str(last.date())
    led["history"].append({"date": str(last.date()), "equity": round(equity, 2)})

    current = {t: q * px[t] / equity for t, q in led["shares"].items()}
    weights = agent.decide(last, closes, current, cfg.max_position)
    led["pending"] = {"decided": str(last.date()), "weights": weights}
    return led


def _portfolios(universe):
    """name -> (agent, columns it may trade)."""
    return {
        "momentum": (MomentumAgent(), universe),
        "stx40": (BuyAndHoldAgent(BENCHMARK), [BENCHMARK]),
        "equal_weight": (HoldAgent(universe), universe),
    }


def _load(name: str, cfg: BacktestConfig) -> dict:
    path = PAPER_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else new_ledger(name, cfg)


def _report(ledgers: dict) -> str:
    bench = ledgers["stx40"]["history"]
    lines = [f"# JSE paper trading - as of {bench[-1]['date']} close", "",
             f"Started {ledgers['stx40']['start']} with R{ledgers['stx40']['capital']:,.0f} each. "
             "Virtual money; no real orders. Not financial advice.", "",
             "| Portfolio | Equity (R) | Return | vs STX40 | Trades | Costs (R) |",
             "|---|---:|---:|---:|---:|---:|"]
    b_ret = bench[-1]["equity"] / ledgers["stx40"]["capital"] - 1
    for name, led in ledgers.items():
        eq = led["history"][-1]["equity"]
        ret = eq / led["capital"] - 1
        costs = sum(t["cost"] for t in led["trades"])
        lines.append(f"| {name} | {eq:,.0f} | {ret:+.2%} | {ret - b_ret:+.2%} | "
                     f"{len(led['trades'])} | {costs:,.0f} |")
    mom = ledgers["momentum"]["pending"]
    if mom:
        picks = ", ".join(f"{t} {w:.0%}" for t, w in sorted(mom["weights"].items())) or "all cash"
        lines += ["", f"Momentum target from the {mom['decided']} close "
                  f"(filled at the next open on the next run): {picks}"]
    return "\n".join(lines) + "\n"


def main():
    from data import load_prices
    cfg = BacktestConfig()
    universe = list(DEFAULT_UNIVERSE)
    opens, closes = load_prices(universe + [BENCHMARK], HISTORY_START, refresh=True)
    now = pd.Timestamp(datetime.now().astimezone())
    opens, closes = completed_bars(opens, now), completed_bars(closes, now)
    universe = [t for t in universe if t in closes.columns]

    PAPER_DIR.mkdir(exist_ok=True)
    ledgers = {}
    for name, (agent, cols) in _portfolios(universe).items():
        led = step(_load(name, cfg), opens[cols], closes[cols], agent, cfg)
        (PAPER_DIR / f"{name}.json").write_text(json.dumps(led, indent=2))
        ledgers[name] = led
    report = _report(ledgers)
    (PAPER_DIR / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
