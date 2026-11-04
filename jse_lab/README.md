# JSE Lab

Research and backtesting for South African (JSE) shares, built alongside Agentic Trading Lab.
Execution is manual: `signals.py` suggests trades, you place them yourself on EasyEquities.

## Setup
    cd jse_lab
    python -m venv .venv
    # macOS/Linux: source .venv/bin/activate    Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env   # only needed for the LLM agent

## Use
    python run_backtest.py --agent momentum
    python run_backtest.py --agent llm --start 2024-01-01 --rebalance 10   # costs API credits
    cp holdings.example.json holdings.json   # then edit to match your EasyEquities account
    python signals.py --agent momentum

Fees in `config.py` are estimates; check them against EasyEquities' current fee schedule.
Backtest results are not a guarantee of future returns.
