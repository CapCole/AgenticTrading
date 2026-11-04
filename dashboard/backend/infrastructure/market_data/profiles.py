"""Canonical market profiles for selectable backtest data sources."""

from __future__ import annotations

from dataclasses import dataclass
import math

from dashboard.backend.infrastructure.llm.validator import DJIA_30
from dashboard.backend.infrastructure.market_data.frequency import (
    TradingFrequency,
    normalize_bar_timeframe,
    normalize_decision_frequency,
)


ALPACA = "alpaca"
VNPY_SIMULATION = "vnpy_simulation"
IFIND_ASHARE = "ifind_ashare"
YAHOO_JSE = "yahoo_jse"
#: Sources whose universe is fixed by the profile, never chosen by the caller.
PROFILE_OWNED_UNIVERSE_SOURCES = ("ifind_ashare", YAHOO_JSE)

RULE_BASED_DECISION_SOURCE = "rule_based"
LLM_DECISION_SOURCE = "llm"

A_SHARE_DEMO_6 = "a_share_demo_6"
A_SHARE_DEMO_6_SYMBOLS = (
    "600519.SH",
    "601318.SH",
    "600036.SH",
    "000001.SZ",
    "000858.SZ",
    "300750.SZ",
)

CSI300_SAMPLE_20_2026H2 = "csi300_sample_20_2026h2"
CSI300_SAMPLE_20_2026H2_SYMBOLS = (
    "600519.SH",
    "601318.SH",
    "600036.SH",
    "300750.SZ",
    "000333.SZ",
    "002594.SZ",
    "600276.SH",
    "300760.SZ",
    "688981.SH",
    "002415.SZ",
    "601766.SH",
    "600309.SH",
    "601899.SH",
    "601857.SH",
    "600900.SH",
    "600050.SH",
    "000725.SZ",
    "600030.SH",
    "600887.SH",
    "600048.SH",
)


@dataclass(frozen=True)
class TransactionCostProfile:
    """Deterministic transaction-cost rules for one market profile.

    Rates are stored as decimal fractions (for example, ``0.00025`` means
    0.025%). The execution layer converts monetary results to Decimal before
    applying currency rounding; keeping the profile immutable makes a run's
    cost assumptions safe to share across its providers and executors.
    """

    version: str
    currency: str
    commission_rate: float
    minimum_commission: float
    stamp_duty_sell_rate: float
    transfer_fee_rate: float
    buy_slippage_rate: float
    sell_slippage_rate: float
    price_tick: float
    #: Tax charged on buys (JSE Securities Transfer Tax). Zero elsewhere.
    stamp_duty_buy_rate: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise ValueError("transaction cost profile version must be non-empty")
        if not str(self.currency).strip():
            raise ValueError("transaction cost profile currency must be non-empty")
        for name in (
            "commission_rate",
            "minimum_commission",
            "stamp_duty_sell_rate",
            "stamp_duty_buy_rate",
            "transfer_fee_rate",
            "buy_slippage_rate",
            "sell_slippage_rate",
            "price_tick",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.price_tick <= 0:
            raise ValueError("price_tick must be positive")

    def to_metadata(self) -> dict[str, object]:
        """Return a stable JSON-safe representation for run metadata."""
        metadata: dict[str, object] = {
            "version": self.version,
            "currency": self.currency,
            "commission_rate": self.commission_rate,
            "minimum_commission": self.minimum_commission,
            "stamp_duty_sell_rate": self.stamp_duty_sell_rate,
            "transfer_fee_rate": self.transfer_fee_rate,
            "buy_slippage_rate": self.buy_slippage_rate,
            "sell_slippage_rate": self.sell_slippage_rate,
            "price_tick": self.price_tick,
        }
        # Only when set: runs persisted before the field existed stay identical.
        if self.stamp_duty_buy_rate:
            metadata["stamp_duty_buy_rate"] = self.stamp_duty_buy_rate
        return metadata


ASHARE_TRANSACTION_COST_PROFILE = TransactionCostProfile(
    version="cn-ashare-default-2026-08",
    currency="CNY",
    commission_rate=0.00025,
    minimum_commission=5.0,
    stamp_duty_sell_rate=0.0005,
    transfer_fee_rate=0.00001,
    buy_slippage_rate=0.0005,
    sell_slippage_rate=0.0005,
    price_tick=0.01,
)


#: EasyEquities-style JSE costs, rands. Estimates; check against the broker's
#: current fee schedule. Commission is 0.25% plus 15% VAT, STT is 0.25% on
#: buys, levies are ~0.01% both sides, slippage 0.1% each way.
JSE_TRANSACTION_COST_PROFILE = TransactionCostProfile(
    version="za-jse-easyequities-estimate-2026-11",
    currency="ZAR",
    commission_rate=0.002875,
    minimum_commission=0.0,
    stamp_duty_sell_rate=0.0,
    stamp_duty_buy_rate=0.0025,
    transfer_fee_rate=0.0001,
    buy_slippage_rate=0.001,
    sell_slippage_rate=0.001,
    price_tick=0.01,
)

JSE_TOP20 = "jse_top20"
#: Liquid JSE large caps on Yahoo Finance (``.JO`` suffix).
JSE_TOP20_SYMBOLS = (
    "NPN.JO", "PRX.JO", "FSR.JO", "SBK.JO", "CPI.JO", "ABG.JO", "NED.JO",
    "MTN.JO", "VOD.JO", "SOL.JO", "AGL.JO", "GLN.JO", "SHP.JO", "BTI.JO",
    "CFR.JO", "GFI.JO", "ANG.JO", "IMP.JO", "BVT.JO", "WHL.JO",
)


@dataclass(frozen=True)
class MarketProfile:
    """Backtest market behavior selected together with a data source."""

    data_source: str
    market: str
    universe: str
    timezone: str
    timeframe: str
    symbols: tuple[str, ...]
    benchmark: str
    default_decision_source: str
    allowed_decision_sources: tuple[str, ...]
    index_baseline_enabled: bool
    native_currency: str
    reporting_currency: str
    t_plus_one_enabled: bool
    lot_size: int = 1
    transaction_cost_profile: TransactionCostProfile | None = None
    # ``timeframe`` remains the backward-compatible decision-bar field.  The
    # fields below make the source, decision, execution and valuation clocks
    # explicit for minute-data work without breaking existing callers.
    source_timeframe: str = "60m"
    decision_frequency: str = "1h"
    execution_timeframe: str = "60m"
    valuation_frequency: str = "60m"

    def __post_init__(self) -> None:
        decision_timeframe = normalize_bar_timeframe(self.timeframe)
        contract = TradingFrequency(
            source_timeframe=self.source_timeframe,
            decision_timeframe=decision_timeframe,
            decision_frequency=self.decision_frequency,
            execution_timeframe=self.execution_timeframe,
            valuation_frequency=self.valuation_frequency,
        )
        object.__setattr__(self, "timeframe", decision_timeframe)
        object.__setattr__(self, "source_timeframe", contract.source_timeframe)
        object.__setattr__(self, "decision_frequency", normalize_decision_frequency(contract.decision_frequency))
        object.__setattr__(self, "execution_timeframe", contract.execution_timeframe)
        object.__setattr__(self, "valuation_frequency", contract.valuation_frequency)

    @property
    def decision_timeframe(self) -> str:
        """Canonical name for the legacy ``timeframe`` decision-bar field."""
        return self.timeframe

    @property
    def frequency_contract(self) -> TradingFrequency:
        """Return the validated frequency contract for this profile."""
        return TradingFrequency(
            source_timeframe=self.source_timeframe,
            decision_timeframe=self.decision_timeframe,
            decision_frequency=self.decision_frequency,
            execution_timeframe=self.execution_timeframe,
            valuation_frequency=self.valuation_frequency,
        )

    @property
    def decision_source(self) -> str:
        """Backward-compatible name for the profile's default decision source."""
        return self.default_decision_source

    @property
    def llm_enabled(self) -> bool:
        """Preserve whether legacy requests default to the LLM path."""
        return self.default_decision_source == LLM_DECISION_SOURCE


_DJIA_30_SYMBOLS = tuple(DJIA_30)

_MARKET_PROFILES = {
    (ALPACA, "djia_30"): MarketProfile(
        data_source=ALPACA,
        market="US",
        universe="djia_30",
        timezone="US/Eastern",
        timeframe="60m",
        symbols=_DJIA_30_SYMBOLS,
        benchmark="top10_buyhold_and_djia",
        default_decision_source=LLM_DECISION_SOURCE,
        allowed_decision_sources=(
            RULE_BASED_DECISION_SOURCE,
            LLM_DECISION_SOURCE,
        ),
        index_baseline_enabled=True,
        native_currency="USD",
        reporting_currency="USD",
        t_plus_one_enabled=False,
        source_timeframe="5m",
        execution_timeframe="5m",
        valuation_frequency="5m",
    ),
    (VNPY_SIMULATION, "djia_30"): MarketProfile(
        data_source=VNPY_SIMULATION,
        market="US",
        universe="djia_30",
        timezone="US/Eastern",
        timeframe="60m",
        symbols=_DJIA_30_SYMBOLS,
        benchmark="top10_buyhold_and_djia",
        default_decision_source=RULE_BASED_DECISION_SOURCE,
        allowed_decision_sources=(RULE_BASED_DECISION_SOURCE,),
        index_baseline_enabled=True,
        native_currency="USD",
        reporting_currency="USD",
        t_plus_one_enabled=False,
    ),
    (IFIND_ASHARE, A_SHARE_DEMO_6): MarketProfile(
        data_source=IFIND_ASHARE,
        market="CN",
        universe=A_SHARE_DEMO_6,
        timezone="Asia/Shanghai",
        timeframe="60m",
        symbols=A_SHARE_DEMO_6_SYMBOLS,
        benchmark="equal_weight_buyhold",
        default_decision_source=LLM_DECISION_SOURCE,
        allowed_decision_sources=(
            RULE_BASED_DECISION_SOURCE,
            LLM_DECISION_SOURCE,
        ),
        index_baseline_enabled=False,
        native_currency="CNY",
        reporting_currency="USD",
        t_plus_one_enabled=True,
        lot_size=100,
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
    ),
    (IFIND_ASHARE, CSI300_SAMPLE_20_2026H2): MarketProfile(
        data_source=IFIND_ASHARE,
        market="CN",
        universe=CSI300_SAMPLE_20_2026H2,
        timezone="Asia/Shanghai",
        timeframe="60m",
        symbols=CSI300_SAMPLE_20_2026H2_SYMBOLS,
        benchmark="equal_weight_buyhold",
        default_decision_source=LLM_DECISION_SOURCE,
        allowed_decision_sources=(
            RULE_BASED_DECISION_SOURCE,
            LLM_DECISION_SOURCE,
        ),
        index_baseline_enabled=False,
        native_currency="CNY",
        reporting_currency="USD",
        t_plus_one_enabled=True,
        lot_size=100,
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
    ),
    (YAHOO_JSE, JSE_TOP20): MarketProfile(
        data_source=YAHOO_JSE,
        market="ZA",
        universe=JSE_TOP20,
        timezone="Africa/Johannesburg",
        timeframe="1d",
        symbols=JSE_TOP20_SYMBOLS,
        benchmark="equal_weight_buyhold",
        default_decision_source=RULE_BASED_DECISION_SOURCE,
        allowed_decision_sources=(
            RULE_BASED_DECISION_SOURCE,
            LLM_DECISION_SOURCE,
        ),
        index_baseline_enabled=False,
        native_currency="ZAR",
        reporting_currency="USD",
        t_plus_one_enabled=False,
        transaction_cost_profile=JSE_TRANSACTION_COST_PROFILE,
        source_timeframe="1d",
        decision_frequency="1d",
        execution_timeframe="1d",
        valuation_frequency="1d",
    ),
}


_DEFAULT_UNIVERSES = {
    ALPACA: "djia_30",
    VNPY_SIMULATION: "djia_30",
    IFIND_ASHARE: A_SHARE_DEMO_6,
    YAHOO_JSE: JSE_TOP20,
}


def registered_market_timezones() -> dict[str, str]:
    """``{market: timezone}`` across the registered profiles.

    A market is one timezone: two profiles pairing the same market with
    different zones would make a dataset keyed on the market ambiguous, so
    that is refused here rather than resolved by whichever profile came first.
    """
    zones: dict[str, str] = {}
    for profile in _MARKET_PROFILES.values():
        existing = zones.setdefault(profile.market, profile.timezone)
        if existing != profile.timezone:
            raise ValueError(
                f"market {profile.market!r} is registered with two timezones: "
                f"{existing!r} and {profile.timezone!r}"
            )
    return zones


def get_market_profile(
    data_source: str,
    universe: str | None = None,
) -> MarketProfile:
    """Return the immutable profile for a registered source/universe pair."""
    resolved_universe = universe or _DEFAULT_UNIVERSES.get(data_source)
    try:
        return _MARKET_PROFILES[(data_source, resolved_universe)]
    except KeyError as exc:
        available = sorted(
            registered_universe
            for registered_source, registered_universe in _MARKET_PROFILES
            if registered_source == data_source
        )
        raise ValueError(
            "Unknown market profile: "
            f"data_source={data_source!r}, universe={resolved_universe!r}, "
            f"available_universes={available!r}"
        ) from exc


def resolve_decision_source(
    profile: MarketProfile,
    requested: str | None = None,
) -> str:
    """Resolve and validate a decision source against one market profile."""
    resolved = (
        str(requested).strip().lower()
        if requested is not None
        else profile.default_decision_source
    )
    if resolved not in profile.allowed_decision_sources:
        raise ValueError(
            "Unsupported decision source: "
            f"data_source={profile.data_source!r}, "
            f"universe={profile.universe!r}, decision_source={resolved!r}, "
            f"allowed_decision_sources={list(profile.allowed_decision_sources)!r}"
        )
    return resolved
