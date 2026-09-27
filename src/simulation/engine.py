"""Simulation engine — drives a paper trade from entry to exit."""

import numpy as np
from dataclasses import dataclass

from src.simulation.broker import PaperBroker, Position
from src.simulation.risk import RiskManager
from src.core.strategies import Strategy
from src.core.pricing import black_scholes


@dataclass
class Frame:
    step: int
    spot: float
    remaining_dte: float
    pnl: float              # scaled
    delta: float
    gamma: float
    theta: float
    vega: float
    leg_prices: list[float] # per unit


class SimulationEngine:
    def __init__(self, broker: PaperBroker, risk: RiskManager,
                 steps: int = 100, annual_drift: float = 0.0):
        self.broker = broker
        self.risk = risk
        self.steps = steps
        self.drift = annual_drift

    def run_with_history(self, strategy: Strategy, spot: float,
                         iv: float, dte: float):
        position = self.broker.enter(strategy, spot, iv, dte)
        history: list[Frame] = []

        dt_days = dte / self.steps
        dt_years = dt_days / 365.0
        current_spot = spot

        history.append(Frame(
            step=0, spot=spot, remaining_dte=dte, pnl=0.0,
            delta=position.entry_greeks.delta,
            gamma=position.entry_greeks.gamma,
            theta=position.entry_greeks.theta,
            vega=position.entry_greeks.vega,
            leg_prices=list(position.entry_leg_prices),
        ))

        for step in range(1, self.steps + 1):
            z = np.random.normal(0.0, 1.0)
            current_spot = current_spot * np.exp(
                (self.drift - 0.5 * iv ** 2) * dt_years
                + iv * np.sqrt(dt_years) * z
            )
            remaining_dte = dte - step * dt_days
            self.broker.mark_to_market(position, current_spot, remaining_dte, iv)

            delta = gamma = theta = vega = 0.0
            T = max(remaining_dte, 0.0) / 365.0
            for leg in strategy.legs:
                sign = +1 if leg.side == "buy" else -1
                _, g = black_scholes(current_spot, leg.strike, T,
                                     self.broker.r, iv, leg.kind)
                delta += sign * leg.qty * g.delta
                gamma += sign * leg.qty * g.gamma
                theta += sign * leg.qty * g.theta
                vega  += sign * leg.qty * g.vega

            history.append(Frame(
                step=step, spot=current_spot, remaining_dte=remaining_dte,
                pnl=position.pnl, delta=delta, gamma=gamma,
                theta=theta, vega=vega,
                leg_prices=list(position.current_leg_prices),
            ))

            exit_reason = self.risk.should_exit(position, remaining_dte)
            if exit_reason is not None:
                position.status = f"CLOSED_{exit_reason}"
                return position, history

        position.status = "CLOSED_TIME"
        return position, history