"""LLM agent tests with a fake Claude client: no API key, no network, no cost.

Run: python -m unittest test_agents
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from agents import LLMAgent, MAX_TOKENS

TICKERS = ["AAA.JO", "BBB.JO", "CCC.JO"]


def _closes(days=260):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=days)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (days, len(TICKERS))), axis=0))
    return pd.DataFrame(walk, index=idx, columns=TICKERS)


class FakeClient:
    """Stands in for anthropic.Anthropic(); replies with queued responses."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        stop_reason, text = self.replies.pop(0)
        content = [SimpleNamespace(type="text", text=text)] if text else []
        return SimpleNamespace(stop_reason=stop_reason, content=content)


def _answer(weights):
    return json.dumps({"weights": weights, "reasoning": "test"})


class LLMAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.closes = _closes()
        self.date = self.closes.index[-1]

    def _agent(self, client, **kw):
        return LLMAgent(client=client, log_dir=self.tmp, **kw)

    def test_complete_answer_is_mapped_back_to_tickers_and_cached(self):
        # ASSET_00 is the alphabetically first ticker, AAA.JO.
        client = FakeClient(("end_turn", _answer({"ASSET_00": 0.15})))
        agent = self._agent(client)

        first = agent.decide(self.date, self.closes, {}, 0.2)
        again = agent.decide(self.date, self.closes, {}, 0.2)

        self.assertEqual(first, {"AAA.JO": 0.15})
        self.assertEqual(again, first)
        self.assertEqual(len(client.requests), 1, "second identical decision should come from cache")
        self.assertEqual(agent.stats, {"api_calls": 1, "cache_hits": 1, "incomplete": 0})

    def test_cut_off_answer_goes_to_cash_is_counted_and_never_cached(self):
        client = FakeClient(("max_tokens", ""), ("end_turn", _answer({"ASSET_01": 0.1})))
        agent = self._agent(client)

        cut_off = agent.decide(self.date, self.closes, {}, 0.2)
        retried = agent.decide(self.date, self.closes, {}, 0.2)

        self.assertEqual(cut_off, {})
        self.assertEqual(retried, {"BBB.JO": 0.1})
        self.assertEqual(len(client.requests), 2, "a cut-off answer must not be replayed from cache")
        self.assertEqual(agent.stats["incomplete"], 1)
        log = [json.loads(line) for line in (self.tmp / "llm_decisions.jsonl").read_text().splitlines()]
        self.assertEqual(log[0]["stop_reason"], "max_tokens")

    def test_refusal_is_treated_like_a_cut_off_answer(self):
        agent = self._agent(FakeClient(("refusal", "")))
        self.assertEqual(agent.decide(self.date, self.closes, {}, 0.2), {})
        self.assertEqual(agent.stats["incomplete"], 1)

    def test_request_leaves_room_for_thinking_sets_effort_and_fallback(self):
        client = FakeClient(("end_turn", _answer({})))
        self._agent(client, effort="low").decide(self.date, self.closes, {}, 0.2)

        request = client.requests[0]
        self.assertEqual(request["max_tokens"], MAX_TOKENS)
        self.assertGreaterEqual(MAX_TOKENS, 8000)
        self.assertEqual(request["output_config"], {"effort": "low"})
        self.assertEqual(request["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", request["betas"])

    def test_changing_effort_does_not_reuse_another_effort_s_cached_answer(self):
        client = FakeClient(("end_turn", _answer({"ASSET_00": 0.1})), ("end_turn", _answer({"ASSET_02": 0.1})))
        self._agent(client, effort="low").decide(self.date, self.closes, {}, 0.2)
        high = self._agent(client, effort="high").decide(self.date, self.closes, {}, 0.2)

        self.assertEqual(high, {"CCC.JO": 0.1})
        self.assertEqual(len(client.requests), 2)

    def test_unknown_effort_is_rejected(self):
        with self.assertRaises(ValueError):
            self._agent(FakeClient(), effort="extreme")


if __name__ == "__main__":
    unittest.main()
