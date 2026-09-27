from dataclasses import dataclass, field
from src.core.pricing import black_scholes, Greeks
from src.core.strategies import Strategy, Leg


# Lot sizes per instrument (as of SEBI Jan 2026 revision)
LOT_SIZES = {
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 60,
    "MIDCPNIFTY": 120,
    "NIFTYNXT50": 25,
}


@dataclass
class Position:
    """A live paper-trade position."""
    strategy: Strategy
    entry_spot: float
    entry_iv: float
    entry_dte: float
    entry_premium: float              # NET premium per unit (signed)
    entry_premium_scaled: float       # NET premium × lot size
    entry_greeks: Greeks
    lot_size: int
    entry_leg_prices: list[float] = field(default_factory=list)     # per unit
    current_leg_prices: list[float] = field(default_factory=list)   # per unit
    pnl: float = 0.0                  # scaled (₹)
    status: str = "OPEN"


class PaperBroker:
    """Fake broker: tracks entry premium, values positions at new spots."""

    def __init__(self, risk_free_rate: float = 0.06, lot_size: int = 65):
        self.r = risk_free_rate
        self.lot_size = lot_size

    def enter(self, strategy: Strategy, spot: float, iv: float, dte: float) -> Position:
        premium = 0.0
        delta = gamma = theta = vega = 0.0
        T = dte / 365.0
        leg_prices = []

        for leg in strategy.legs:
            price, g = black_scholes(spot, leg.strike, T, self.r, iv, leg.kind)
            sign = +1 if leg.side == "buy" else -1
            premium += sign * leg.qty * price
            delta   += sign * leg.qty * g.delta
            gamma   += sign * leg.qty * g.gamma
            theta   += sign * leg.qty * g.theta
            vega    += sign * leg.qty * g.vega
            leg_prices.append(price)

        return Position(
            strategy=strategy,
            entry_spot=spot,
            entry_iv=iv,
            entry_dte=dte,
            entry_premium=premium,
            entry_premium_scaled=premium * self.lot_size,
            entry_greeks=Greeks(delta, gamma, theta, vega),
            lot_size=self.lot_size,
            entry_leg_prices=leg_prices,
            current_leg_prices=list(leg_prices),
        )

    def mark_to_market(self, position: Position, spot: float,
                       remaining_dte: float, iv: float) -> float:
        if position.status != "OPEN":
            return position.pnl

        current_value = 0.0
        current_prices = []
        T = max(remaining_dte, 0.0) / 365.0

        for leg in position.strategy.legs:
            price, _ = black_scholes(spot, leg.strike, T, self.r, iv, leg.kind)
            sign = +1 if leg.side == "buy" else -1
            current_value += sign * leg.qty * price
            current_prices.append(price)

        position.current_leg_prices = current_prices
        # PnL per unit × lot size = scaled ₹
        pnl_per_unit = current_value - position.entry_premium
        position.pnl = pnl_per_unit * position.lot_size
        return position.pnl