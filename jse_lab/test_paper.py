"""Forward paper-trading ledger tests on synthetic prices. No network.

Run: python -m unittest test_paper
"""
import unittest

import pandas as pd

from config import BacktestConfig
from paper import HoldAgent, completed_bars, new_ledger, step

DAYS = pd.bdate_range("2026-11-02", periods=6)  # Mon 2 Nov .. Mon 9 Nov


def _frames(open_px=100.0, close_px=100.0, tickers=("AAA", "BBB")):
    opens = pd.DataFrame(open_px, index=DAYS, columns=list(tickers))
    closes = pd.DataFrame(close_px, index=DAYS, columns=list(tickers))
    return opens, closes


class FixedAgent:
    def __init__(self, weights):
        self.weights = weights

    def decide(self, date, closes, holdings, max_position):
        return dict(self.weights)


class StepTest(unittest.TestCase):
    def setUp(self):
        self.cfg = BacktestConfig(capital=100_000.0)

    def test_first_run_only_decides_and_trades_nothing(self):
        opens, closes = _frames()
        led = step(new_ledger("t", self.cfg), opens.iloc[:3], closes.iloc[:3],
                   FixedAgent({"AAA": 0.5}), self.cfg)
        self.assertEqual(led["cash"], 100_000.0)
        self.assertEqual(led["trades"], [])
        self.assertEqual(led["pending"], {"decided": "2026-11-04", "weights": {"AAA": 0.5}})
        self.assertEqual(led["history"], [{"date": "2026-11-04", "equity": 100_000.0}])

    def test_pending_fills_at_the_next_days_open_with_costs(self):
        opens, closes = _frames()
        opens.loc[DAYS[3], "AAA"] = 80.0  # the day after the decision opens at 80
        agent = FixedAgent({"AAA": 0.5})
        led = step(new_ledger("t", self.cfg), opens.iloc[:3], closes.iloc[:3], agent, self.cfg)
        led = step(led, opens, closes, agent, self.cfg)
        trade = led["trades"][0]
        self.assertEqual(trade["date"], "2026-11-05")
        self.assertEqual(trade["ticker"], "AAA")
        self.assertAlmostEqual(trade["value"], 50_000.0)
        self.assertAlmostEqual(led["shares"]["AAA"], 50_000.0 / 80.0)
        cost = self.cfg.costs.cost(50_000.0, True)
        self.assertAlmostEqual(trade["cost"], round(cost, 2))
        self.assertAlmostEqual(led["cash"], 50_000.0 - cost)

    def test_original_ledger_is_not_mutated(self):
        opens, closes = _frames()
        led0 = new_ledger("t", self.cfg)
        step(led0, opens, closes, FixedAgent({"AAA": 0.5}), self.cfg)
        self.assertIsNone(led0["pending"])
        self.assertEqual(led0["history"], [])

    def test_rerun_on_the_same_close_changes_nothing(self):
        opens, closes = _frames()
        led = step(new_ledger("t", self.cfg), opens, closes, FixedAgent({"AAA": 0.5}), self.cfg)
        again = step(led, opens, closes, FixedAgent({"BBB": 1.0}), self.cfg)
        self.assertEqual(again, led)

    def test_pending_waits_when_no_later_bar_exists(self):
        opens, closes = _frames()
        led = step(new_ledger("t", self.cfg), opens, closes, FixedAgent({"AAA": 0.5}), self.cfg)
        self.assertEqual(led["trades"], [])
        self.assertIsNotNone(led["pending"])

    def test_sells_happen_before_buys(self):
        opens, closes = _frames()
        cfg = BacktestConfig(capital=100_000.0, max_position=1.0)
        led = step(new_ledger("t", cfg), opens.iloc[:2], closes.iloc[:2], FixedAgent({"AAA": 1.0}), cfg)
        led = step(led, opens.iloc[:4], closes.iloc[:4], FixedAgent({"BBB": 1.0}), cfg)
        led = step(led, opens, closes, FixedAgent({"BBB": 1.0}), cfg)
        day = [t["ticker"] for t in led["trades"] if t["date"] == "2026-11-06"]
        self.assertEqual(day, ["AAA", "BBB"])  # the sale funds the purchase
        self.assertGreater(led["shares"]["BBB"], 0)


class HoldAgentTest(unittest.TestCase):
    def test_buys_equal_weights_once_then_holds(self):
        _, closes = _frames()
        agent = HoldAgent(["AAA", "BBB"])
        self.assertEqual(agent.decide(DAYS[0], closes, {}, 1.0), {"AAA": 0.5, "BBB": 0.5})
        held = {"AAA": 0.6, "BBB": 0.4}
        self.assertEqual(agent.decide(DAYS[1], closes, held, 1.0), held)


class CompletedBarsTest(unittest.TestCase):
    def test_drops_todays_bar_before_the_jse_close(self):
        _, closes = _frames()
        before = pd.Timestamp("2026-11-09 15:00", tz="Africa/Johannesburg")
        after = pd.Timestamp("2026-11-09 18:00", tz="Africa/Johannesburg")
        self.assertEqual(completed_bars(closes, before).index[-1], DAYS[4])
        self.assertEqual(completed_bars(closes, after).index[-1], DAYS[5])


if __name__ == "__main__":
    unittest.main()
