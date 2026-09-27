"""
Plotly chart builders for the dashboard.

All functions return a plotly.graph_objects.Figure.
The UI layer just renders them with st.plotly_chart().
"""

import numpy as np
import plotly.graph_objects as go
from src.core.strategies import Strategy
from src.core.pricing import black_scholes
from src.simulation.engine import Frame


def payoff_figure(
    strategy: Strategy,
    entry_spot: float,
    entry_iv: float,
    entry_dte: float,
    r: float,
    current_spot: float | None = None,
    current_pnl: float | None = None,
) -> go.Figure:
    """
    Payoff curve at expiry. X-axis = spot price, Y-axis = PnL.
    Optionally overlays a marker for the current spot during animation.
    """
    # Build a spot range around entry spot: +/-20%
    spot_min = entry_spot * 0.80
    spot_max = entry_spot * 1.20
    spots = np.linspace(spot_min, spot_max, 200)

    # Payoff at expiry = sum of leg intrinsic values (no premium yet)
    payoff = np.zeros_like(spots)
    for leg in strategy.legs:
        sign = +1 if leg.side == "buy" else -1
        if leg.kind == "call":
            intrinsic = np.maximum(spots - leg.strike, 0.0)
        else:
            intrinsic = np.maximum(leg.strike - spots, 0.0)
        payoff += sign * leg.qty * intrinsic

    # Subtract entry premium so the curve shows actual PnL
    T = entry_dte / 365.0
    premium = 0.0
    for leg in strategy.legs:
        sign = +1 if leg.side == "buy" else -1
        price, _ = black_scholes(entry_spot, leg.strike, T, r, entry_iv, leg.kind)
        premium += sign * leg.qty * price
    pnl_at_expiry = payoff - premium

    fig = go.Figure()

    # Profit region (green)
    fig.add_trace(go.Scatter(
        x=spots, y=np.maximum(pnl_at_expiry, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(0, 200, 0, 0.15)",
        name="Profit", showlegend=False,
    ))

    # Loss region (red)
    fig.add_trace(go.Scatter(
        x=spots, y=np.minimum(pnl_at_expiry, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(220, 0, 0, 0.15)",
        name="Loss", showlegend=False,
    ))

    # Main payoff line
    fig.add_trace(go.Scatter(
        x=spots, y=pnl_at_expiry,
        mode="lines", line=dict(color="white", width=2),
        name="PnL at expiry",
    ))

    # Zero line (breakeven)
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))

    # Current spot marker (animated during playback)
    if current_spot is not None and current_pnl is not None:
        fig.add_trace(go.Scatter(
            x=[current_spot], y=[current_pnl],
            mode="markers+text",
            marker=dict(color="yellow", size=14, symbol="diamond"),
            text=[f"  Spot: {current_spot:.2f}<br>  PnL: {current_pnl:.2f}"],
            textposition="top right",
            name="Current", showlegend=False,
        ))

    fig.update_layout(
        title=f"{strategy.name}",
        xaxis_title="Spot price",
        yaxis_title="PnL",
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def pnl_path_figure(history: list[Frame]) -> go.Figure:
    """PnL over the course of the simulation."""
    steps = [f.step for f in history]
    pnls = [f.pnl for f in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=pnls, mode="lines",
        line=dict(color="cyan", width=2),
        name="PnL",
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title="PnL over time",
        xaxis_title="Step",
        yaxis_title="PnL",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def spot_path_figure(history: list[Frame]) -> go.Figure:
    """Spot price over the course of the simulation."""
    steps = [f.step for f in history]
    spots = [f.spot for f in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=spots, mode="lines",
        line=dict(color="orange", width=2),
        name="Spot",
    ))
    fig.update_layout(
        title="Spot price over time",
        xaxis_title="Step",
        yaxis_title="Spot",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig