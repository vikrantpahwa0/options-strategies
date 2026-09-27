"""
Runner: execute a single paper trade and print the result.

Usage:
    python -m src.simulation.run
"""

from src.core.strategies import condor
from src.simulation.broker import PaperBroker
from src.simulation.risk import RiskManager
from src.simulation.engine import SimulationEngine


def main():
    # 1. Build a Nifty condor: spot ~25000, 100-point wings.
    strategy = condor(
        low=24800, mid_low=24900, mid_high=25100, high=25200, kind="call"
    )

    # 2. Set up the paper broker and risk manager.
    broker = PaperBroker(risk_free_rate=0.06)
    risk = RiskManager(profit_target_pct=0.50, stop_loss_pct=2.0, exit_dte=1.0)

    # 3. Create the simulation engine.
    engine = SimulationEngine(broker=broker, risk=risk, steps=100)

    # 4. Run the trade.
    position = engine.run(
        strategy=strategy,
        spot=25000.0,
        iv=0.15,
        dte=7.0,
    )

    # 5. Report.
    print(f"Strategy:      {position.strategy.name}")
    print(f"Entry spot:    {position.entry_spot:.2f}")
    print(f"Entry premium: {position.entry_premium:.2f}")
    print(f"Entry Greeks:  delta={position.entry_greeks.delta:.3f} "
          f"gamma={position.entry_greeks.gamma:.5f} "
          f"theta={position.entry_greeks.theta:.3f} "
          f"vega={position.entry_greeks.vega:.3f}")
    print(f"Final PnL:     {position.pnl:.2f}")
    print(f"Status:        {position.status}")


if __name__ == "__main__":
    main()