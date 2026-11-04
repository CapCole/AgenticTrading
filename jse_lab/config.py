"""Settings for the JSE lab. Verify the fee figures against EasyEquities' current fee schedule."""
from dataclasses import dataclass, field

# Liquid JSE large caps (Yahoo Finance uses the .JO suffix). Tickers with no data are skipped.
DEFAULT_UNIVERSE = [
    "NPN.JO", "PRX.JO", "FSR.JO", "SBK.JO", "CPI.JO", "ABG.JO", "NED.JO",
    "MTN.JO", "VOD.JO", "SOL.JO", "AGL.JO", "GLN.JO", "SHP.JO", "BTI.JO",
    "CFR.JO", "GFI.JO", "ANG.JO", "IMP.JO", "BVT.JO", "WHL.JO",
]
BENCHMARK = "STX40.JO"  # Satrix 40 ETF, a buy-and-hold JSE Top 40 baseline


@dataclass
class CostModel:
    brokerage_pct: float = 0.0025      # broker commission per trade
    vat_on_brokerage: float = 0.15     # SA VAT charged on the commission
    stt_on_buys_pct: float = 0.0025    # Securities Transfer Tax, buys only
    other_levies_pct: float = 0.0001   # investor protection levy, settlement etc. (approx.)
    slippage_pct: float = 0.0010       # spread + market impact; raise for small caps

    def cost(self, trade_value: float, is_buy: bool) -> float:
        v = abs(trade_value)
        c = v * self.brokerage_pct * (1 + self.vat_on_brokerage)
        c += v * self.other_levies_pct + v * self.slippage_pct
        if is_buy:
            c += v * self.stt_on_buys_pct
        return c


@dataclass
class BacktestConfig:
    start: str = "2019-01-01"
    end: str | None = None
    capital: float = 100_000.0          # rands
    rebalance_every: int = 5            # trading days (5 = weekly)
    max_position: float = 0.20          # cap on any single share
    min_trade_value: float = 500.0      # skip tiny trades that cost more than they help
    risk_free_rate: float = 0.07        # approx. SA cash rate, for the Sharpe ratio
    costs: CostModel = field(default_factory=CostModel)
