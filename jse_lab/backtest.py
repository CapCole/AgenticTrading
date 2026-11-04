"""Walk-forward backtest: decide at day t's close, trade at day t+1's open."""
import numpy as np
import pandas as pd

from config import BacktestConfig


def run_backtest(agent, opens, closes, cfg: BacktestConfig, warmup=210):
    dates = closes.index
    cash, shares = cfg.capital, {t: 0.0 for t in closes.columns}
    pending, equity_curve, trades = None, [], []
    total_costs = 0.0

    for i, date in enumerate(dates):
        # 1) Execute yesterday's decision at today's open (fractional shares allowed).
        if pending is not None:
            px = opens.loc[date]
            equity = cash + sum(shares[t] * px[t] for t in shares if not np.isnan(px[t]))
            orders = []
            for t in shares:
                if np.isnan(px[t]):
                    continue
                diff = pending.get(t, 0.0) * equity - shares[t] * px[t]
                if abs(diff) >= cfg.min_trade_value:
                    orders.append((t, diff))
            for t, diff in sorted(orders, key=lambda o: o[1]):  # sells first, frees cash
                is_buy = diff > 0
                cost = cfg.costs.cost(diff, is_buy)
                if is_buy:
                    diff = min(diff, cash - cost)
                    if diff <= 0:
                        continue
                shares[t] += diff / px[t]
                cash -= diff + cost
                total_costs += cost
                trades.append({"date": date, "ticker": t, "value": round(diff, 2),
                               "cost": round(cost, 2)})
            pending = None

        # 2) Mark to market at the close.
        px = closes.loc[date]
        equity = cash + sum(shares[t] * px[t] for t in shares if not np.isnan(px[t]))
        equity_curve.append((date, equity))

        # 3) Decide on rebalance days, using only data up to today.
        if i >= warmup and (i - warmup) % cfg.rebalance_every == 0 and i + 1 < len(dates):
            current = {t: shares[t] * px[t] / equity for t in shares if shares[t] > 0}
            pending = agent.decide(date, closes, current, cfg.max_position)

    curve = pd.Series(dict(equity_curve)).iloc[warmup:]
    return curve, pd.DataFrame(trades), total_costs


def metrics(curve: pd.Series, rf=0.0) -> dict:
    rets = curve.pct_change().dropna()
    years = len(rets) / 252
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol > 0 else float("nan")
    mdd = (curve / curve.cummax() - 1).min()
    return {"final_equity": round(float(curve.iloc[-1]), 2),
            "total_return": round(float(curve.iloc[-1] / curve.iloc[0] - 1), 4),
            "cagr": round(float(cagr), 4), "volatility": round(float(vol), 4),
            "sharpe": round(float(sharpe), 2), "max_drawdown": round(float(mdd), 4)}
