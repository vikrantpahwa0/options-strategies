"""
Risk manager: decides when to exit a position.

Rules (all optional):
- Stop-loss:     exit when PnL <= -sl_pct × |entry_premium|. Capped at max loss.
- Profit target: exit when PnL >= tp_pct × basis. Basis is either
                 "premium" (entry premium paid/received) or
                 "max_profit" (theoretical max profit of the position).
- Time exit:     exit when remaining DTE <= exit_dte.

Priority order: SL first (most critical), then TP, then TIME.
"""

from dataclasses import dataclass
from src.simulation.broker import Position


@dataclass
class RiskManager:
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 2.00
    exit_dte: float = 1.0
    tp_basis: str = "premium"       # "premium" or "max_profit"
    max_profit: float = 0.0         # ₹ value, set by caller (0 = ignore)

    def should_exit(self, position: Position, remaining_dte: float) -> str | None:
        """
        Return exit reason string or None.
        Reasons: 'SL' (stop-loss), 'TP' (target), 'TIME'.
        """
        premium_scaled = abs(position.entry_premium_scaled)

        # --- Compute TP based on chosen basis ---
        if self.tp_basis == "max_profit" and self.max_profit > 0:
            tp_value = self.profit_target_pct * self.max_profit
        else:
            tp_value = self.profit_target_pct * premium_scaled

        # --- Compute SL, capped at max loss ---
        # On debit strategies, max loss = premium paid. So SL can't exceed that.
        sl_value = self.stop_loss_pct * premium_scaled
        sl_value = min(sl_value, premium_scaled)

        # --- Check order: SL, TP, TIME ---
        if sl_value > 0 and position.pnl <= -sl_value:
            return "SL"
        if tp_value > 0 and position.pnl >= tp_value:
            return "TP"
        if remaining_dte <= self.exit_dte:
            return "TIME"
        return None