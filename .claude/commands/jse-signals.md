Check this week's JSE trade suggestions from the JSE Lab. Do not place any orders.

1. From jse_lab/ with its .venv active, run `python signals.py --agent momentum` (or the agent I name: $ARGUMENTS). Ask me first if the agent is `claude`; it costs API credits.
2. Read results/latest_signals.json.
3. For each suggested trade, show the features behind it (momentum, volatility, distance from moving averages, drawdown, RSI) and flag anything that stands out: concentration in one sector (e.g. several miners or banks), very high volatility, trades too small to justify their costs, or prices that look stale or wrong.
4. Estimate the total trading costs for the suggested trades using the cost model in config.py.
5. Finish with a short table: share, action, rand amount, estimated cost, and any flags. Describe what the agent's rules and the data show; don't tell me whether to make the trade.

Remind me that I place trades manually on EasyEquities, that the decision is mine, and that this is not financial advice.
