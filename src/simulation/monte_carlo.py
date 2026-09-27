"""
Monte Carlo runner: execute N simulations of the same strategy
with different random paths, collect statistics.
"""

from dataclasses import dataclass, field
import numpy as np

from src.simulation.broker import PaperBroker
from src.simulation.risk import RiskManager
from src.simulation.engine import SimulationEngine
from src.core.strategies import Strategy


@dataclass
class MonteCarloResult:
    n_runs: int
    pnls: list[float]
    statuses: list[str]

    @property
    def avg_pnl(self) -> float:
        return float(np.mean(self.pnls))

    @property
    def median_pnl(self) -> float:
        return float(np.median(self.pnls))

    @property
    def best_pnl(self) -> float:
        return float(np.max(self.pnls))

    @property
    def worst_pnl(self) -> float:
        return float(np.min(self.pnls))

    @property
    def std_pnl(self) -> float:
        return float(np.std(self.pnls))

    @property
    def win_rate(self) -> float:
        return sum(1 for s in self.statuses if s == "CLOSED_TP") / self.n_runs

    def status_counts(self) -> dict[str, int]:
        counts = {}
        for s in self.statuses:
            counts[s] = counts.get(s, 0) + 1
        return counts


def run_monte_carlo(
    strategy: Strategy,
    spot: float,
    iv: float,
    dte: float,
    lot_size: int,
    tp_pct: float,
    sl_pct: float,
    tp_basis: str,
    max_profit: float,
    n_runs: int = 100,
    steps: int = 100,
    seed: int | None = None,
) -> MonteCarloResult:
    """
    Run the same strategy `n_runs` times, each with a fresh random path.
    Returns aggregated stats.
    """
    if seed is not None:
        np.random.seed(seed)

    pnls = []
    statuses = []

    for _ in range(n_runs):
        broker = PaperBroker(risk_free_rate=0.06, lot_size=lot_size)
        risk = RiskManager(
            profit_target_pct=tp_pct,
            stop_loss_pct=sl_pct,
            exit_dte=1.0,
            tp_basis=tp_basis,
            max_profit=max_profit,
        )
        engine = SimulationEngine(broker=broker, risk=risk, steps=steps)
        final_position, _ = engine.run_with_history(strategy, spot, iv, dte)
        pnls.append(final_position.pnl)
        statuses.append(final_position.status)

    return MonteCarloResult(n_runs=n_runs, pnls=pnls, statuses=statuses)