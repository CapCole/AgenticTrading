"""Agents return target portfolio weights. Anything not allocated stays in cash."""
import hashlib
import json
import os
import re
import warnings
from pathlib import Path

from features import snapshot

try:  # load ANTHROPIC_API_KEY from jse_lab/.env if present
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

LOG_DIR = Path(__file__).parent / "logs"


def clean_weights(raw: dict, universe, max_position: float) -> dict:
    """Validate agent output: known tickers, no shorts, position cap, total <= 100%."""
    w = {}
    for t, v in (raw or {}).items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if t in universe and v > 0:
            w[t] = min(v, max_position)
    total = sum(w.values())
    if total > 1.0:
        w = {t: v / total for t, v in w.items()}
    return w


class BuyAndHoldAgent:
    name = "buy_and_hold"

    def __init__(self, ticker):
        self.ticker = ticker

    def decide(self, date, closes, holdings, max_position):
        return {self.ticker: 1.0}


class MomentumAgent:
    """Baseline: hold the top N shares by 6-month return that sit above their 200-day average."""
    name = "momentum"

    def __init__(self, top_n=5):
        self.top_n = top_n

    def decide(self, date, closes, holdings, max_position):
        feats = snapshot(closes, date)
        ok = {t: f for t, f in feats.items() if f["vs_ma200"] > 0}
        ranked = sorted(ok, key=lambda t: ok[t]["ret_6m"], reverse=True)[: self.top_n]
        if not ranked:
            return {}
        return clean_weights({t: 1.0 / len(ranked) for t in ranked}, closes.columns, max_position)


SYSTEM_PROMPT = """You are a cautious long-only portfolio manager for South African (JSE) shares.
You receive technical features for each asset. Decide target portfolio weights.
Rules:
- Weights are fractions of total equity, each between 0 and {max_position}.
- Total weight must be <= 1.0; the rest stays in cash. Holding cash is allowed.
- Every trade costs roughly 0.6% round trip, so avoid needless turnover.
- Only use assets listed in the input.
Respond with ONLY a JSON object, no markdown:
{{"weights": {{"ASSET": 0.1}}, "reasoning": "one or two sentences"}}"""


#: Room for Claude's thinking plus the JSON answer. Thinking counts toward this cap,
#: so 1,000 tokens could be used up before any answer was written.
MAX_TOKENS = 8000
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "medium"  # override with CLAUDE_EFFORT in .env
#: On a policy decline, the API re-runs the request on a model it picks (server-side).
FALLBACK_BETA = "server-side-fallback-2026-07-01"
COMPLETE = "end_turn"


class LLMAgent:
    """Claude-powered agent. Complete decisions are cached on disk so reruns don't cost API calls.

    anonymize=True hides ticker names and dates. Claude was trained on historical prices,
    so in a backtest it might recognise "Naspers in March 2020" and use hindsight.
    Keep anonymize on for backtests; it's optional for live signals.

    An answer that was cut off or declined is logged and counted in ``stats["incomplete"]``,
    the agent holds cash for that period, and the answer is not cached, so a rerun asks again.
    """
    name = "llm"

    def __init__(self, model=None, anonymize=True, run_name="llm", effort=None,
                 client=None, log_dir=LOG_DIR):
        self.effort = effort or os.getenv("CLAUDE_EFFORT", DEFAULT_EFFORT)
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {self.effort!r}")
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
        self.anonymize = anonymize
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        self.cache_path = log_dir / "llm_cache.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.log_path = log_dir / f"{run_name}_decisions.jsonl"
        self.stats = {"api_calls": 0, "cache_hits": 0, "incomplete": 0}

    def _alias(self, tickers):
        if not self.anonymize:
            return {t: t for t in tickers}
        return {t: f"ASSET_{i:02d}" for i, t in enumerate(sorted(tickers))}

    def _ask(self, system, user):
        """One Claude call. Returns (text, stop_reason)."""
        self.stats["api_calls"] += 1
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA], fallbacks="default",
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.stop_reason

    def decide(self, date, closes, holdings, max_position):
        feats = snapshot(closes, date)
        if not feats:
            return {}
        alias = self._alias(feats.keys())
        back = {a: t for t, a in alias.items()}
        payload = {
            "date": None if self.anonymize else str(date.date()),
            "current_weights": {alias[t]: round(w, 4) for t, w in holdings.items() if t in alias},
            "assets": {alias[t]: f for t, f in feats.items()},
        }
        system = SYSTEM_PROMPT.format(max_position=max_position)
        user = json.dumps(payload)
        key = hashlib.sha256((self.model + self.effort + system + user).encode()).hexdigest()

        if key in self.cache:
            self.stats["cache_hits"] += 1
            text, stop_reason = self.cache[key], COMPLETE
        else:
            text, stop_reason = self._ask(system, user)
            if stop_reason == COMPLETE:
                self.cache[key] = text
                self.cache_path.write_text(json.dumps(self.cache))
            else:
                self.stats["incomplete"] += 1
                warnings.warn(f"{date.date()}: Claude stopped with {stop_reason!r}; holding cash, not caching")
                text = ""

        parsed = self._parse(text)
        raw = {back.get(a, a): v for a, v in parsed.get("weights", {}).items()}
        weights = clean_weights(raw, closes.columns, max_position)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"date": str(date.date()), "weights": weights, "stop_reason": stop_reason,
                                "reasoning": parsed.get("reasoning", ""), "raw": text}) + "\n")
        return weights

    @staticmethod
    def _parse(text):
        text = re.sub(r"```(?:json)?", "", text).strip()
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}  # unparseable answer -> go to cash rather than guess


def make_agent(name, **kw):
    if name == "momentum":
        return MomentumAgent(top_n=kw.get("top_n", 5))
    if name == "llm":
        return LLMAgent(anonymize=kw.get("anonymize", True), run_name=kw.get("run_name", "llm"),
                        effort=kw.get("effort"))
    raise ValueError(f"Unknown agent: {name}")
