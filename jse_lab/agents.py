"""Agents return target portfolio weights. Anything not allocated stays in cash."""
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import warnings
from pathlib import Path

from features import snapshot

try:  # load settings and ANTHROPIC_API_KEY from jse_lab/.env if present
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
OLLAMA_TIMEOUT_S = 300  # a local 8B model can take a while on a laptop


class _LLMBase:
    """Shared logic for AI agents: build features, call the model, validate, cache, log.

    anonymize=True hides ticker names and dates. Models were trained on historical prices,
    so in a backtest they might recognise "Naspers in March 2020" and use hindsight.
    Keep anonymize on for backtests; it's optional for live signals.

    Only complete answers are cached. An answer that was cut off or declined is logged,
    counted in ``stats["incomplete"]``, and the agent holds cash for that period; a rerun asks again.
    """
    name = "llm"
    complete_reason = ""  # the model's stop reason for a finished answer

    def __init__(self, model, anonymize=True, run_name=None, log_dir=LOG_DIR):
        self.model = model
        self.anonymize = anonymize
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        self.cache_path = log_dir / "llm_cache.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.log_path = log_dir / f"{run_name or self.name}_decisions.jsonl"
        self.stats = {"model_calls": 0, "cache_hits": 0, "incomplete": 0}

    def _complete(self, system, user):
        """One model call. Returns (text, stop_reason)."""
        raise NotImplementedError

    def _cache_variant(self):
        """Settings besides the model that change the answer, so they belong in the cache key."""
        return ""

    def _alias(self, tickers):
        if not self.anonymize:
            return {t: t for t in tickers}
        return {t: f"ASSET_{i:02d}" for i, t in enumerate(sorted(tickers))}

    def _answer(self, date, system, user):
        key = hashlib.sha256("|".join(
            [self.name, self.model, self._cache_variant(), system, user]).encode()).hexdigest()
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key], self.complete_reason
        self.stats["model_calls"] += 1
        text, stop_reason = self._complete(system, user)
        if stop_reason == self.complete_reason:
            self.cache[key] = text
            self.cache_path.write_text(json.dumps(self.cache))
            return text, stop_reason
        self.stats["incomplete"] += 1
        warnings.warn(f"{date.date()}: {self.name} stopped with {stop_reason!r}; holding cash, not caching")
        return "", stop_reason

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
        text, stop_reason = self._answer(date, system, json.dumps(payload))

        parsed = self._parse(text)
        weights_in = parsed.get("weights", {}) if isinstance(parsed.get("weights"), dict) else {}
        raw = {back.get(a, a): v for a, v in weights_in.items()}
        weights = clean_weights(raw, closes.columns, max_position)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"date": str(date.date()), "model": self.model, "weights": weights,
                                "stop_reason": stop_reason, "reasoning": parsed.get("reasoning", ""),
                                "raw": text}) + "\n")
        return weights

    @staticmethod
    def _parse(text):
        text = re.sub(r"```(?:json)?", "", text or "").strip()
        m = re.search(r"\{.*\}", text, re.S)
        try:
            parsed = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}  # unparseable answer -> go to cash rather than guess
        return parsed if isinstance(parsed, dict) else {}


class ClaudeAgent(_LLMBase):
    """Uses the Anthropic API. Needs ANTHROPIC_API_KEY in .env and costs API credits."""
    name = "claude"
    complete_reason = "end_turn"

    def __init__(self, model=None, anonymize=True, run_name=None, effort=None,
                 client=None, log_dir=LOG_DIR):
        self.effort = effort or os.getenv("CLAUDE_EFFORT", DEFAULT_EFFORT)
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {self.effort!r}")
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        super().__init__(model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
                         anonymize, run_name, log_dir)

    def _cache_variant(self):
        return self.effort

    def _complete(self, system, user):
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA], fallbacks="default",
        )
        return "".join(b.text for b in resp.content if b.type == "text"), resp.stop_reason


def _post_json(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as r:
        return json.loads(r.read())


class OllamaAgent(_LLMBase):
    """Free: runs an open model on your own computer through Ollama (https://ollama.com).
    Start Ollama and pull a model first, e.g. `ollama pull llama3.1:8b`."""
    name = "ollama"
    complete_reason = "stop"

    def __init__(self, model=None, anonymize=True, run_name=None, base_url=None,
                 post=_post_json, log_dir=LOG_DIR):
        base = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.url = base.rstrip("/") + "/api/chat"
        self._post = post
        super().__init__(model or os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                         anonymize, run_name, log_dir)

    def _complete(self, system, user):
        body = {
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        try:
            resp = self._post(self.url, body)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Can't reach Ollama at {self.url}. Is the Ollama app running, and have you "
                f"run `ollama pull {self.model}`?") from e
        return resp["message"]["content"], resp.get("done_reason", self.complete_reason)


def make_agent(name, **kw):
    common = {"anonymize": kw.get("anonymize", True), "run_name": kw.get("run_name")}
    if name == "momentum":
        return MomentumAgent(top_n=kw.get("top_n", 5))
    if name == "claude":
        return ClaudeAgent(effort=kw.get("effort"), client=kw.get("client"), **common)
    if name == "ollama":
        return OllamaAgent(post=kw.get("post", _post_json), **common)
    raise ValueError(f"Unknown agent: {name}")
