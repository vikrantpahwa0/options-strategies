from dataclasses import dataclass
from typing import Literal


OptionKind = Literal["call", "put"]
Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Leg:
    """One option position inside a strategy."""
    kind: OptionKind
    side: Side
    strike: float
    qty: int


@dataclass(frozen=True)
class Strategy:
    """A collection of legs with a name."""
    name: str
    legs: list[Leg]


def butterfly(center: float, width: float, kind: OptionKind = "call") -> Strategy:
    """
    Long Butterfly: buy low, sell 2 middle, buy high.
    Bet: underlying ends near `center` at expiry.
    """
    return Strategy(
        name=f"{kind}_butterfly_{center}",
        legs=[
            Leg(kind, "buy",  center - width, 1),
            Leg(kind, "sell", center,         2),
            Leg(kind, "buy",  center + width, 1),
        ],
    )


def condor(
    low: float,
    mid_low: float,
    mid_high: float,
    high: float,
    kind: OptionKind = "call",
) -> Strategy:
    """
    Long Condor: buy low, sell two middles, buy high.
    Bet: underlying stays between mid_low and mid_high at expiry.
    """
    return Strategy(
        name=f"{kind}_condor_{low}_{high}",
        legs=[
            Leg(kind, "buy",  low,      1),
            Leg(kind, "sell", mid_low,  1),
            Leg(kind, "sell", mid_high, 1),
            Leg(kind, "buy",  high,     1),
        ],
    )