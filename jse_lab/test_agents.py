"""AI agent tests with fake Claude and Ollama backends: no API key, no network, no cost.

Run: python -m unittest test_agents
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from agents import MAX_TOKENS, ClaudeAgent, OllamaAgent, make_agent

TICKERS = ["AAA.JO", "BBB.JO", "CCC.JO"]


def _closes(days=260):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=days)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (days, len(TICKERS))), axis=0))
    return pd.DataFrame(walk, index=idx, columns=TICKERS)


def _answer(weights):
    return json.dumps({"weights": weights, "reasoning": "test"})


class FakeClaude:
    """Stands in for anthropic.Anthropic(); replies with queued (stop_reason, text) pairs."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        stop_reason, text = self.replies.pop(0)
        content = [SimpleNamespace(type="text", text=text)] if text else []
        return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeOllama:
    """Stands in for the HTTP POST to Ollama's /api/chat; replies with queued (done_reason, text)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, url, body):
        self.requests.append((url, body))
        done_reason, text = self.replies.pop(0)
        return {"message": {"role": "assistant", "content": text}, "done": True, "done_reason": done_reason}


class _AgentCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.closes = _closes()
        self.date = self.closes.index[-1]

    def decide(self, agent):
        return agent.decide(self.date, self.closes, {}, 0.2)


class ClaudeAgentTest(_AgentCase):
    def _agent(self, client, **kw):
        return ClaudeAgent(client=client, log_dir=self.tmp, **kw)

    def test_complete_answer_is_mapped_back_to_tickers_and_cached(self):
        # ASSET_00 is the alphabetically first ticker, AAA.JO.
        client = FakeClaude(("end_turn", _answer({"ASSET_00": 0.15})))
        agent = self._agent(client)

        first, again = self.decide(agent), self.decide(agent)

        self.assertEqual(first, {"AAA.JO": 0.15})
        self.assertEqual(again, first)
        self.assertEqual(len(client.requests), 1, "second identical decision should come from cache")
        self.assertEqual(agent.stats, {"model_calls": 1, "cache_hits": 1, "incomplete": 0})

    def test_cut_off_answer_goes_to_cash_is_counted_and_never_cached(self):
        client = FakeClaude(("max_tokens", ""), ("end_turn", _answer({"ASSET_01": 0.1})))
        agent = self._agent(client)

        cut_off, retried = self.decide(agent), self.decide(agent)

        self.assertEqual(cut_off, {})
        self.assertEqual(retried, {"BBB.JO": 0.1})
        self.assertEqual(len(client.requests), 2, "a cut-off answer must not be replayed from cache")
        self.assertEqual(agent.stats["incomplete"], 1)
        log = [json.loads(line) for line in (self.tmp / "claude_decisions.jsonl").read_text().splitlines()]
        self.assertEqual(log[0]["stop_reason"], "max_tokens")

    def test_refusal_is_treated_like_a_cut_off_answer(self):
        agent = self._agent(FakeClaude(("refusal", "")))
        self.assertEqual(self.decide(agent), {})
        self.assertEqual(agent.stats["incomplete"], 1)

    def test_request_leaves_room_for_thinking_sets_effort_and_fallback(self):
        client = FakeClaude(("end_turn", _answer({})))
        self.decide(self._agent(client, effort="low"))

        request = client.requests[0]
        self.assertEqual(request["max_tokens"], MAX_TOKENS)
        self.assertGreaterEqual(MAX_TOKENS, 8000)
        self.assertEqual(request["output_config"], {"effort": "low"})
        self.assertEqual(request["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", request["betas"])

    def test_changing_effort_does_not_reuse_another_effort_s_cached_answer(self):
        client = FakeClaude(("end_turn", _answer({"ASSET_00": 0.1})), ("end_turn", _answer({"ASSET_02": 0.1})))
        self.decide(self._agent(client, effort="low"))
        high = self.decide(self._agent(client, effort="high"))

        self.assertEqual(high, {"CCC.JO": 0.1})
        self.assertEqual(len(client.requests), 2)

    def test_unknown_effort_is_rejected(self):
        with self.assertRaises(ValueError):
            self._agent(FakeClaude(), effort="extreme")


class OllamaAgentTest(_AgentCase):
    def _agent(self, post, **kw):
        return OllamaAgent(post=post, log_dir=self.tmp, **kw)

    def test_complete_answer_is_mapped_back_and_cached(self):
        post = FakeOllama(("stop", _answer({"ASSET_02": 0.2})))
        agent = self._agent(post)

        first, again = self.decide(agent), self.decide(agent)

        self.assertEqual(first, {"CCC.JO": 0.2})
        self.assertEqual(again, first)
        self.assertEqual(len(post.requests), 1)
        self.assertEqual(agent.stats, {"model_calls": 1, "cache_hits": 1, "incomplete": 0})

    def test_request_is_deterministic_json_chat_to_the_configured_server(self):
        post = FakeOllama(("stop", _answer({})))
        self.decide(self._agent(post, model="llama3.1:8b", base_url="http://localhost:11434"))

        url, body = post.requests[0]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertEqual(body["model"], "llama3.1:8b")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["format"], "json")
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])

    def test_answer_cut_off_by_length_goes_to_cash_and_is_not_cached(self):
        post = FakeOllama(("length", '{"weights": {"ASSET_0'), ("stop", _answer({"ASSET_00": 0.1})))
        agent = self._agent(post)

        self.assertEqual(self.decide(agent), {})
        self.assertEqual(self.decide(agent), {"AAA.JO": 0.1})
        self.assertEqual(agent.stats["incomplete"], 1)

    def test_garbage_answer_goes_to_cash(self):
        agent = self._agent(FakeOllama(("stop", "I think you should buy everything!")))
        self.assertEqual(self.decide(agent), {})

    def test_ollama_and_claude_answers_never_share_a_cache_entry(self):
        claude = ClaudeAgent(client=FakeClaude(("end_turn", _answer({"ASSET_00": 0.1}))), log_dir=self.tmp)
        self.decide(claude)
        post = FakeOllama(("stop", _answer({"ASSET_01": 0.1})))

        self.assertEqual(self.decide(self._agent(post)), {"BBB.JO": 0.1})
        self.assertEqual(len(post.requests), 1)


class MakeAgentTest(unittest.TestCase):
    def test_names_match_the_cli_choices(self):
        self.assertEqual(make_agent("momentum").name, "momentum")
        self.assertIsInstance(make_agent("ollama", post=FakeOllama()), OllamaAgent)
        self.assertIsInstance(make_agent("claude", client=FakeClaude()), ClaudeAgent)

    def test_unknown_agent_is_rejected(self):
        with self.assertRaises(ValueError):
            make_agent("llm")


if __name__ == "__main__":
    unittest.main()
