"""Agents return target portfolio weights. Anything not allocated stays in cash."""
import hashlib
import json
import os
import re
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


class LLMAgent:
    """Claude-powered agent. Decisions are cached on disk so reruns don't cost API calls.

    anonymize=True hides ticker names and dates. Claude was trained on historical prices,
    so in a backtest it might recognise "Naspers in March 2020" and use hindsight.
    Keep anonymize on for backtests; it's optional for live signals.
    """
    name = "llm"

    def __init__(self, model=None, anonymize=True, run_name="llm"):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
        self.anonymize = anonymize
        LOG_DIR.mkdir(exist_ok=True)
        self.cache_path = LOG_DIR / "llm_cache.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.log_path = LOG_DIR / f"{run_name}_decisions.jsonl"

    def _alias(self, tickers):
        if not self.anonymize:
            return {t: t for t in tickers}
        return {t: f"ASSET_{i:02d}" for i, t in enumerate(sorted(tickers))}

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
        key = hashlib.sha256((self.model + system + user).encode()).hexdigest()

        if key in self.cache:
            text = self.cache[key]
        else:
            resp = self.client.messages.create(
                model=self.model, max_tokens=1000, system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            self.cache[key] = text
            self.cache_path.write_text(json.dumps(self.cache))

        parsed = self._parse(text)
        raw = {back.get(a, a): v for a, v in parsed.get("weights", {}).items()}
        weights = clean_weights(raw, closes.columns, max_position)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"date": str(date.date()), "weights": weights,
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
        return LLMAgent(anonymize=kw.get("anonymize", True), run_name=kw.get("run_name", "llm"))
    raise ValueError(f"Unknown agent: {name}")
