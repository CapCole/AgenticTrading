# JSE Lab

Research and backtesting for South African (JSE) shares, built alongside Agentic Trading Lab.
Execution is manual: `signals.py` suggests trades, you place them yourself on EasyEquities.

## Setup
    cd jse_lab
    python -m venv .venv
    # macOS/Linux: source .venv/bin/activate    Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env

## Agents
| Agent      | Cost                  | Notes                                             |
|------------|-----------------------|---------------------------------------------------|
| `momentum` | Free                  | Rule-based baseline. Every AI agent must beat it. |
| `ollama`   | Free (local)          | Open model on your computer via ollama.com.       |
| `claude`   | Anthropic API credits | Strongest reasoning; needs ANTHROPIC_API_KEY.     |

AI answers are cached in `logs/llm_cache.json`, so rerunning the same backtest is free.
An answer that was cut off is not cached: the agent holds cash for that period and the
metrics report it under `model.incomplete`. If that count isn't 0, the result understates the agent.

## Use
    python run_backtest.py --agent momentum
    python run_backtest.py --agent ollama --start 2024-01-01 --rebalance 10
    cp holdings.example.json holdings.json   # edit to match your EasyEquities account
    python signals.py --agent momentum
Then in Claude Code, run `/jse-signals` for a check of the suggestions (uses your subscription, not API credits).

## Forward paper trading
    python paper.py     # once a week, after the JSE close (17:30 SAST)
Runs three virtual R100k portfolios on live prices: `momentum`, `stx40` (benchmark) and
`equal_weight` (buy once, hold). Each run fills last week's orders at the next day's open with
costs, values everything at the latest close, and sets new targets. Ledgers and `report.md` are in
`paper/`; commit them weekly so the record can't be edited after the fact. No real orders.

## Tests
    python -m unittest test_agents test_data   # no network, no API key, no cost

Fees in `config.py` are estimates; check them against EasyEquities' current fee schedule.
Backtest results are not a guarantee of future returns. Not financial advice.
