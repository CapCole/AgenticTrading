# JSE Lab: context for Claude Code

## Goal
Research South African (JSE) shares with rule-based and AI agents, backtest them honestly
against the Satrix 40 (STX40.JO), and produce trade suggestions the owner places manually
on EasyEquities. Success means beating STX40 after costs, not just making money.

## Hard rules
- Never write code that places real orders on EasyEquities or any broker. EasyEquities has no
  official API; unofficial scrapers are fragile and may breach its terms.
- Never hard-code, print or commit API keys or passwords. Secrets live in jse_lab/.env.
  To check a key is present, print only its length.
- Ask before running `--agent claude` (costs API credits). `momentum` and `ollama` are free.
- Work on the feature/jse-lab branch. Never push to main, never force-push, ask before every push,
  and never open a pull request to the upstream repo.
- Ask before editing anything in dashboard/.

## Layout
- config.py: universe, benchmark, EasyEquities cost model, backtest settings
- data.py: Yahoo Finance .JO prices (cents converted to rands, rands-for-cents glitches rescaled), CSV cache
- features.py: point-in-time features (no lookahead)
- agents.py: MomentumAgent, OllamaAgent (free/local), ClaudeAgent (API); output is validated
- backtest.py: decide at close t, trade at open t+1; metrics
- run_backtest.py: CLI backtests; outputs in results/
- signals.py: today's suggested trades from holdings.json; writes results/latest_signals.json
- test_agents.py, test_data.py: run with `python -m unittest test_agents test_data` (no network)

## AI agent details
- ClaudeAgent uses max_tokens=8000 because adaptive thinking counts toward the cap; 1000 left
  no room for the answer. Effort comes from CLAUDE_EFFORT or --effort (default medium).
- Only complete answers (Claude `end_turn`, Ollama `stop`) are cached. Others hold cash, are not
  cached, and are counted in the metrics under `model.incomplete`. Report that count with results.

## Things to keep honest
- AI agents can recognise historical periods; keep anonymize=True in backtests.
- Include costs and slippage in every comparison. Compare to momentum and STX40.
- Prefer fewer, larger rebalances; JSE trading costs are high relative to small edges.
- Not financial advice. Say so when presenting results.
