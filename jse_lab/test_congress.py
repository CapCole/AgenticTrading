"""Copy-Congress strategy tests on synthetic disclosures. No network.

Run: python -m unittest test_congress
"""
import unittest

import pandas as pd

from congress import CongressCopyAgent, build_lots, normalize_ticker, parse_trades

DAYS = pd.bdate_range("2024-01-01", periods=120)


def _raw(filer, ticker, kind, filed, chamber="house", branch="congress", traded=None):
    return {"filer_id": filer, "ticker": ticker, "transaction_type": kind, "filing_date": filed,
            "transaction_date": traded or filed, "_chamber": chamber, "_branch": branch}


def _closes(tickers=("AAA", "BBB", "CCC")):
    return pd.DataFrame(100.0, index=DAYS, columns=list(tickers))


class ParseTradesTest(unittest.TestCase):
    def test_keeps_congress_stock_buys_and_sales_only(self):
        trades = parse_trades([
            _raw("a", "AAA", "Purchase", "2024-01-10"),
            _raw("a", "AAA", "Sale (Partial)", "2024-02-10"),
            _raw("a", "AAA", "Exchange", "2024-02-11"),
            _raw("x", "BBB", "Purchase", "2024-01-10", chamber=None, branch="executive"),
            _raw("a", None, "Purchase", "2024-01-10"),
            _raw("a", "--", "Purchase", "2024-01-10"),
        ])
        self.assertEqual(list(trades["side"]), ["buy", "sell"])
        self.assertEqual(set(trades["ticker"]), {"AAA"})

    def test_senate_only_filter(self):
        rows = [_raw("s", "AAA", "Purchase", "2024-01-10", chamber="senate"),
                _raw("h", "BBB", "Purchase", "2024-01-10", chamber="house")]
        self.assertEqual(list(parse_trades(rows, chamber="senate")["ticker"]), ["AAA"])

    def test_ticker_normalisation_matches_yahoo(self):
        self.assertEqual(normalize_ticker("brk.b"), "BRK-B")
        self.assertEqual(normalize_ticker(" msft "), "MSFT")
        self.assertIsNone(normalize_ticker("N/A"))
        self.assertIsNone(normalize_ticker("ABCDEFG"))


class LotsTest(unittest.TestCase):
    def test_fixed_hold_ends_after_the_holding_period(self):
        lots = build_lots(parse_trades([_raw("a", "AAA", "Purchase", "2024-01-10")]),
                          hold="fixed", hold_days=60)
        self.assertEqual(lots.iloc[0]["start"], pd.Timestamp("2024-01-10"))
        self.assertEqual(lots.iloc[0]["end"], pd.Timestamp("2024-03-10"))

    def test_until_sale_ends_when_the_same_member_files_a_sale(self):
        trades = parse_trades([
            _raw("a", "AAA", "Purchase", "2024-01-10"),
            _raw("b", "AAA", "Sale (Full)", "2024-01-20"),   # someone else selling: ignored
            _raw("a", "AAA", "Sale (Full)", "2024-02-05"),
        ])
        lots = build_lots(trades, hold="until_sale", hold_days=365)
        self.assertEqual(lots.iloc[0]["end"], pd.Timestamp("2024-02-05"))

    def test_until_sale_is_capped(self):
        lots = build_lots(parse_trades([_raw("a", "AAA", "Purchase", "2024-01-10")]),
                          hold="until_sale", hold_days=30)
        self.assertEqual(lots.iloc[0]["end"], pd.Timestamp("2024-02-09"))


class AgentTest(unittest.TestCase):
    def _agent(self, rows, **kw):
        return CongressCopyAgent(build_lots(parse_trades(rows), **kw))

    def test_a_buy_is_invisible_until_its_filing_date(self):
        # Traded on 2 Jan but filed on 1 Feb: a follower can only act after 1 Feb.
        agent = self._agent([_raw("a", "AAA", "Purchase", "2024-02-01", traded="2024-01-02")])
        self.assertEqual(agent.decide(pd.Timestamp("2024-01-31"), _closes(), {}, 0.2), {})
        self.assertEqual(agent.decide(pd.Timestamp("2024-02-01"), _closes(), {}, 0.2), {"AAA": 0.2})

    def test_position_is_dropped_once_the_lot_ends(self):
        agent = self._agent([_raw("a", "AAA", "Purchase", "2024-01-10")], hold="fixed", hold_days=10)
        self.assertEqual(agent.decide(pd.Timestamp("2024-01-19"), _closes(), {}, 1.0), {"AAA": 1.0})
        self.assertEqual(agent.decide(pd.Timestamp("2024-01-20"), _closes(), {}, 1.0), {})

    def test_active_tickers_share_the_money_equally(self):
        agent = self._agent([_raw("a", "AAA", "Purchase", "2024-01-10"),
                             _raw("b", "BBB", "Purchase", "2024-01-11"),
                             _raw("c", "AAA", "Purchase", "2024-01-12")])
        weights = agent.decide(pd.Timestamp("2024-01-15"), _closes(), {}, 1.0)
        self.assertEqual(weights, {"AAA": 0.5, "BBB": 0.5})

    def test_top_n_keeps_the_most_bought_then_most_recent(self):
        rows = [_raw("a", "AAA", "Purchase", "2024-01-10"), _raw("b", "AAA", "Purchase", "2024-01-11"),
                _raw("a", "BBB", "Purchase", "2024-01-09"), _raw("c", "CCC", "Purchase", "2024-01-12")]
        agent = CongressCopyAgent(build_lots(parse_trades(rows)), top_n=2)
        # AAA has two open buys; CCC beats BBB on the more recent filing.
        self.assertEqual(agent.decide(pd.Timestamp("2024-01-15"), _closes(), {}, 1.0),
                         {"AAA": 0.5, "CCC": 0.5})

    def test_tickers_without_a_price_are_skipped(self):
        agent = self._agent([_raw("a", "ZZZ", "Purchase", "2024-01-10")])
        self.assertEqual(agent.decide(pd.Timestamp("2024-01-15"), _closes(), {}, 1.0), {})


if __name__ == "__main__":
    unittest.main()
