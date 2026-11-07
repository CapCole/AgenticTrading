"""Today's suggested trades, for you to review and place manually on EasyEquities.

Keep holdings.json up to date yourself, e.g.:
  {"cash": 25000, "shares": {"NPN.JO": 3.5, "SBK.JO": 40}}

Usage: python signals.py --agent momentum
"""
import argparse
import json
from pathlib import Path

from agents import make_agent
from config import DEFAULT_UNIVERSE, BacktestConfig
from data import load_prices
from features import snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="momentum", choices=["momentum", "ollama", "claude"])
    p.add_argument("--holdings", default="holdings.json")
    a = p.parse_args()

    cfg = BacktestConfig()
    h = json.loads(Path(a.holdings).read_text()) if Path(a.holdings).exists() else {"cash": cfg.capital, "shares": {}}
    _, closes = load_prices(DEFAULT_UNIVERSE, "2023-01-01", refresh=True)
    date, px = closes.index[-1], closes.iloc[-1]

    values = {t: q * px[t] for t, q in h["shares"].items() if t in px}
    equity = h["cash"] + sum(values.values())
    current = {t: v / equity for t, v in values.items()}

    agent = make_agent(a.agent, anonymize=False, run_name=f"{a.agent}_live")
    target = agent.decide(date, closes, current, cfg.max_position)

    print(f"Prices as of {date.date()} close. Portfolio value R{equity:,.0f}\n")
    print(f"{'Share':<10}{'Now':>8}{'Target':>8}{'Trade (R)':>14}")
    for t in sorted(set(current) | set(target)):
        diff = target.get(t, 0) * equity - values.get(t, 0)
        if abs(diff) < cfg.min_trade_value:
            continue
        side = "BUY" if diff > 0 else "SELL"
        print(f"{t:<10}{current.get(t, 0):>8.1%}{target.get(t, 0):>8.1%}{diff:>14,.0f}  {side}")
    out = Path("results")
    out.mkdir(exist_ok=True)
    report = {"as_of": str(date.date()), "agent": a.agent, "equity": round(equity, 2),
              "current_weights": current, "target_weights": target,
              "features": snapshot(closes, date)}
    (out / "latest_signals.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved full details to {(out / 'latest_signals.json').resolve()}")
    print("These are suggestions only. Check them before placing any order.")


if __name__ == "__main__":
    main()
