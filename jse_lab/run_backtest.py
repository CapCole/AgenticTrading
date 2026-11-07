"""Usage:
  python run_backtest.py --agent momentum
  python run_backtest.py --agent ollama --start 2024-01-01 --rebalance 10   (free, local)
  python run_backtest.py --agent claude --start 2024-01-01 --rebalance 10   (API credits)
"""
import argparse
import json
from pathlib import Path

from agents import BuyAndHoldAgent, make_agent
from backtest import metrics, run_backtest
from config import BENCHMARK, DEFAULT_UNIVERSE, BacktestConfig
from data import load_prices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="momentum", choices=["momentum", "ollama", "claude"])
    p.add_argument("--tickers", nargs="*", default=DEFAULT_UNIVERSE)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--rebalance", type=int, default=5, help="trading days between decisions")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--no-anonymize", action="store_true")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                   help="Claude thinking effort (default: CLAUDE_EFFORT or medium)")
    p.add_argument("--refresh", action="store_true", help="re-download prices")
    a = p.parse_args()

    cfg = BacktestConfig(start=a.start, end=a.end, capital=a.capital, rebalance_every=a.rebalance)
    tickers = list(dict.fromkeys(a.tickers + [BENCHMARK]))
    opens, closes = load_prices(tickers, a.start, a.end, refresh=a.refresh)

    universe = [t for t in closes.columns if t != BENCHMARK]
    agent = make_agent(a.agent, top_n=a.top_n, anonymize=not a.no_anonymize, run_name=a.agent,
                       effort=a.effort)
    curve, trades, costs = run_backtest(agent, opens[universe], closes[universe], cfg)

    results = {"agent": metrics(curve, cfg.risk_free_rate)}
    results["agent"]["total_costs"] = round(costs, 2)
    results["agent"]["num_trades"] = len(trades)
    if hasattr(agent, "stats"):  # AI agents: model calls, cache hits, cut-off or declined answers
        results["agent"]["model"] = {"name": agent.model, **agent.stats}
    if BENCHMARK in closes.columns:
        bh, _, _ = run_backtest(BuyAndHoldAgent(BENCHMARK), opens[[BENCHMARK]],
                                closes[[BENCHMARK]], cfg)
        bh = bh.loc[curve.index[0]:]
        results["benchmark_STX40"] = metrics(bh, cfg.risk_free_rate)

    out = Path("results")
    out.mkdir(exist_ok=True)
    curve.to_csv(out / f"{a.agent}_equity.csv")
    trades.to_csv(out / f"{a.agent}_trades.csv", index=False)
    (out / f"{a.agent}_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved equity curve, trades and metrics to {out.resolve()}")


if __name__ == "__main__":
    main()
