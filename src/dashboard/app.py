"""Streamlit dashboard for the Butterfly & Condor Lab."""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.core.strategies import butterfly, condor
from src.simulation.broker import PaperBroker, LOT_SIZES
from src.simulation.risk import RiskManager
from src.simulation.engine import SimulationEngine
from src.simulation.monte_carlo import run_monte_carlo
from src.dashboard.charts import payoff_figure, pnl_path_figure, spot_path_figure




st.set_page_config(page_title="Butterfly & Condor Lab", layout="wide")
st.title("Butterfly & Condor Lab")
st.caption("Paper-trading simulator — Black-Scholes pricing + Greeks + automatic exits")


# ----- Sidebar -----
with st.sidebar:
    st.header("Instrument")
    instrument = st.selectbox("Index", list(LOT_SIZES.keys()))
    lot_size = LOT_SIZES[instrument]
    st.caption(f"Lot size: {lot_size} units per contract")

    st.header("Strategy")
    strategy_type = st.selectbox("Type", ["Condor", "Butterfly"])
    center = st.number_input("Center strike", value=25000.0, step=100.0)
    width = st.number_input("Width", value=200.0, step=50.0)

    st.header("Market")
    spot = st.number_input("Entry spot", value=25000.0, step=50.0)
    iv = st.slider("Implied Volatility", 0.05, 0.60, 0.15, 0.01)
    dte = st.number_input("Days to expiry", value=7.0, min_value=1.0, step=1.0)

    st.header("Risk")
    tp_mode = st.selectbox(
        "Profit target basis",
        ["% of premium", "% of max profit"],
    )
    tp_pct = st.slider("Profit target", 0.10, 2.00, 0.50, 0.05)
    sl_pct = st.slider("Stop-loss (x premium)", 0.10, 5.00, 0.50, 0.10)

    st.header("Simulation")
    steps = st.slider("Steps", 20, 200, 100, 10)
    speed = st.slider("Animation speed (ms/frame)", 50, 500, 150, 25)

    run_button = st.button("Run Simulation", type="primary")


def build_strategy():
    if strategy_type == "Butterfly":
        return butterfly(center=center, width=width, kind="call")
    return condor(
        low=center - 2 * width, mid_low=center - width,
        mid_high=center + width, high=center + 2 * width, kind="call",
    )


def compute_position_metrics(strategy, entry_spot, iv, dte, lot_size):
    broker = PaperBroker(risk_free_rate=0.06, lot_size=lot_size)
    position = broker.enter(strategy, entry_spot, iv, dte)
    net_premium = position.entry_premium_scaled
    entry_premium_unit = position.entry_premium

    spots = np.linspace(entry_spot * 0.7, entry_spot * 1.3, 1000)
    payoffs = np.zeros_like(spots)
    for leg in strategy.legs:
        sign = +1 if leg.side == "buy" else -1
        if leg.kind == "call":
            intrinsic = np.maximum(spots - leg.strike, 0.0)
        else:
            intrinsic = np.maximum(leg.strike - spots, 0.0)
        payoffs += sign * leg.qty * intrinsic

    pnl_curve = (payoffs - entry_premium_unit) * lot_size
    max_profit = float(pnl_curve.max())
    max_loss = float(pnl_curve.min())

    breakevens = []
    for i in range(1, len(spots)):
        if pnl_curve[i - 1] * pnl_curve[i] < 0:
            breakevens.append(round(spots[i], 1))

    return {
        "net_premium": net_premium,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakevens": breakevens,
    }


def render_leg_table(strategy, entry_prices, current_prices, lot_size):
    rows = []
    net_entry = 0.0
    net_current = 0.0

    for i, leg in enumerate(strategy.legs):
        entry = entry_prices[i]
        current = current_prices[i]
        sign = +1 if leg.side == "buy" else -1
        entry_cash = -sign * entry * leg.qty * lot_size
        current_cash = -sign * current * leg.qty * lot_size
        leg_pnl = entry_cash - current_cash

        net_entry += entry_cash
        net_current += current_cash

        rows.append({
            "Leg": f"{leg.side.upper()} {leg.kind.upper()} {leg.strike}",
            "Qty": leg.qty,
            "Entry (per unit)": round(entry, 2),
            "Entry (₹)": round(entry_cash, 2),
            "Current (per unit)": round(current, 2),
            "Current (₹)": round(current_cash, 2),
            "Leg PnL (₹)": round(leg_pnl, 2),
        })

    rows.append({
        "Leg": "NET / TOTAL",
        "Qty": "",
        "Entry (per unit)": "",
        "Entry (₹)": round(net_entry, 2),
        "Current (per unit)": "",
        "Current (₹)": round(net_current, 2),
        "Leg PnL (₹)": round(net_entry - net_current, 2),
    })
    return rows


tab_payoff, tab_live, tab_mc = st.tabs(
    ["Payoff", "Live Simulation", "Monte Carlo"]
)


# ----- Tab 1: Payoff -----
with tab_payoff:
    strategy = build_strategy()
    fig = payoff_figure(strategy=strategy, entry_spot=spot,
                        entry_iv=iv, entry_dte=dte, r=0.06)
    st.plotly_chart(fig, use_container_width=True)

    metrics = compute_position_metrics(strategy, spot, iv, dte, lot_size)
    st.subheader(f"Position summary ({instrument}, lot={lot_size})")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Net premium (₹)", f"₹{metrics['net_premium']:,.2f}")
    s2.metric("Max profit (₹)", f"₹{metrics['max_profit']:,.2f}")
    s3.metric("Max loss (₹)", f"₹{metrics['max_loss']:,.2f}")
    s4.metric("Breakevens",
              " / ".join(f"{b:,.0f}" for b in metrics["breakevens"]) or "—")

    broker = PaperBroker(risk_free_rate=0.06, lot_size=lot_size)
    position = broker.enter(strategy, spot, iv, dte)

    st.subheader("Strategy legs — at entry")
    rows = render_leg_table(strategy, position.entry_leg_prices,
                            position.entry_leg_prices, lot_size)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Greeks at entry")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta", f"{position.entry_greeks.delta:.4f}")
    g2.metric("Gamma", f"{position.entry_greeks.gamma:.5f}")
    g3.metric("Theta", f"{position.entry_greeks.theta:.4f}")
    g4.metric("Vega", f"{position.entry_greeks.vega:.4f}")

    st.subheader("Exit rules")
    premium_scaled = abs(metrics["net_premium"])
    if tp_mode == "% of max profit":
        tp_base = abs(metrics["max_profit"])
    else:
        tp_base = premium_scaled
    tp_value = tp_pct * tp_base
    sl_value = min(sl_pct * premium_scaled, premium_scaled)

    e1, e2, e3 = st.columns(3)
    e1.metric("Profit target", f"₹{tp_value:,.2f}",
              help=f"{tp_pct*100:.0f}% of ₹{tp_base:,.2f} ({tp_mode})")
    e2.metric("Stop-loss", f"−₹{sl_value:,.2f}",
              help=f"{sl_pct:.2f}× of premium, capped at max loss")
    e3.metric("Time exit", "DTE ≤ 1")

    if sl_pct * premium_scaled > premium_scaled:
        st.warning(
            f"Stop-loss of {sl_pct:.2f}× exceeds max loss "
            f"(₹{premium_scaled:,.2f}). It has been capped."
        )


# ----- Tab 2: Live -----
with tab_live:
    st.write("Click **Run Simulation** in the sidebar.")

    # ---- Single animated run ----
    if run_button:
        strategy = build_strategy()
        metrics = compute_position_metrics(strategy, spot, iv, dte, lot_size)

        broker = PaperBroker(risk_free_rate=0.06, lot_size=lot_size)
        risk = RiskManager(
            profit_target_pct=tp_pct, stop_loss_pct=sl_pct, exit_dte=1.0,
            tp_basis="max_profit" if tp_mode == "% of max profit" else "premium",
            max_profit=abs(metrics["max_profit"]),
        )
        engine = SimulationEngine(broker=broker, risk=risk, steps=steps)
        final_position, history = engine.run_with_history(
            strategy=strategy, spot=spot, iv=iv, dte=dte,
        )

        premium_scaled = abs(final_position.entry_premium_scaled)
        tp_base = abs(metrics["max_profit"]) if tp_mode == "% of max profit" else premium_scaled
        tp_value = tp_pct * tp_base
        sl_value = min(sl_pct * premium_scaled, premium_scaled)

        st.subheader(f"Live playback — {instrument} (lot={lot_size})")
        metrics_placeholder = st.empty()
        thresholds_placeholder = st.empty()
        legs_placeholder = st.empty()
        chart_placeholder = st.empty()

        for i, frame in enumerate(history):
            prev = history[i - 1] if i > 0 else frame

            with metrics_placeholder.container():
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Spot", f"{frame.spot:,.2f}",
                          delta=f"{frame.spot - prev.spot:+,.2f}")
                m2.metric("PnL (₹)", f"₹{frame.pnl:,.2f}",
                          delta=f"₹{frame.pnl - prev.pnl:+,.2f}")
                m3.metric("Delta", f"{frame.delta:.4f}")
                m4.metric("Theta", f"{frame.theta:.4f}")
                m5.metric("Step", f"{frame.step}/{steps}")

            with thresholds_placeholder.container():
                pnl = frame.pnl
                tp_progress = max(0, min(100, (pnl / tp_value) * 100)) if tp_value > 0 else 0
                sl_buffer = -sl_value - pnl
                t1, t2, t3 = st.columns(3)
                t1.metric("Profit target", f"₹{tp_value:,.2f}",
                          delta=f"{tp_progress:.0f}% reached")
                t2.metric("Stop-loss", f"−₹{sl_value:,.2f}",
                          delta=f"₹{abs(sl_buffer):,.2f} buffer" if sl_buffer < 0 else "breached")
                t3.metric("Max loss", f"−₹{abs(metrics['max_loss']):,.2f}")

            with legs_placeholder.container():
                rows = render_leg_table(
                    strategy, final_position.entry_leg_prices,
                    frame.leg_prices, lot_size,
                )
                st.dataframe(rows, use_container_width=True, hide_index=True)

            if i % 5 == 0 or i == len(history) - 1:
                fig = payoff_figure(
                    strategy=strategy, entry_spot=spot,
                    entry_iv=iv, entry_dte=dte, r=0.06,
                    current_spot=frame.spot, current_pnl=frame.pnl,
                )
                with chart_placeholder:
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"payoff_{i}")

            time.sleep(speed / 1000.0)

        steps_used = len(history) - 1
        if final_position.status == "CLOSED_TP":
            st.success(f"**CLOSED_TP** — Target: ₹{tp_value:,.2f} | "
                       f"Final: ₹{final_position.pnl:,.2f} | Steps: {steps_used}/{steps}")
        elif final_position.status == "CLOSED_SL":
            slippage = abs(final_position.pnl) - sl_value
            st.error(f"**CLOSED_SL** — Threshold: −₹{sl_value:,.2f} | "
                     f"Filled: ₹{final_position.pnl:,.2f} | "
                     f"Slippage: ₹{max(slippage, 0):,.2f} | Steps: {steps_used}/{steps}")
        elif final_position.status == "CLOSED_TIME":
            st.warning(f"**CLOSED_TIME** — Final: ₹{final_position.pnl:,.2f} | "
                       f"Steps: {steps_used}/{steps}")
        else:
            st.info(f"Status: {final_position.status} | PnL: ₹{final_position.pnl:,.2f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(pnl_path_figure(history), use_container_width=True)
        with col_b:
            st.plotly_chart(spot_path_figure(history), use_container_width=True)

    # ---- Animate N, then stream remaining runs into a live-updating table ----
    st.divider()
    st.subheader("Animate + Stream")

    c1, c2, c3 = st.columns(3)
    n_animate = c1.slider("Runs to animate", 1, 10, 5, 1)
    n_total = c2.slider("Total runs", 10, 200, 100, 10)
    run_stream = c3.button("Run Animate + Stream", type="primary", key="run_stream")

    if run_stream:
        strategy = build_strategy()
        metrics = compute_position_metrics(strategy, spot, iv, dte, lot_size)

        premium_scaled = abs(metrics["net_premium"])
        tp_base = abs(metrics["max_profit"]) if tp_mode == "% of max profit" else premium_scaled
        tp_value = tp_pct * tp_base
        sl_value = min(sl_pct * premium_scaled, premium_scaled)

        # ----- Phase 1: Animate the first N runs -----
        st.markdown(f"### Phase 1 — Animating {n_animate} runs")
        animated_results = []

        for run_i in range(1, n_animate + 1):
            st.markdown(f"**Run {run_i} / {n_animate}**")

            broker = PaperBroker(risk_free_rate=0.06, lot_size=lot_size)
            risk = RiskManager(
                profit_target_pct=tp_pct, stop_loss_pct=sl_pct, exit_dte=1.0,
                tp_basis="max_profit" if tp_mode == "% of max profit" else "premium",
                max_profit=abs(metrics["max_profit"]),
            )
            engine = SimulationEngine(broker=broker, risk=risk, steps=steps)
            final_position, history = engine.run_with_history(
                strategy=strategy, spot=spot, iv=iv, dte=dte,
            )

            run_metrics_ph = st.empty()
            run_chart_ph = st.empty()

            for j, frame in enumerate(history):
                prev = history[j - 1] if j > 0 else frame
                with run_metrics_ph.container():
                    rm1, rm2, rm3, rm4 = st.columns(4)
                    rm1.metric("Spot", f"{frame.spot:,.2f}",
                               delta=f"{frame.spot - prev.spot:+,.2f}")
                    rm2.metric("PnL (₹)", f"₹{frame.pnl:,.2f}",
                               delta=f"₹{frame.pnl - prev.pnl:+,.2f}")
                    rm3.metric("Step", f"{frame.step}/{steps}")
                    rm4.metric("Status",
                               final_position.status if j == len(history)-1 else "running")

                if j % 5 == 0 or j == len(history) - 1:
                    fig = payoff_figure(
                        strategy=strategy, entry_spot=spot,
                        entry_iv=iv, entry_dte=dte, r=0.06,
                        current_spot=frame.spot, current_pnl=frame.pnl,
                    )
                    with run_chart_ph:
                        st.plotly_chart(fig, use_container_width=True,
                                        key=f"anim_{run_i}_{j}")

                time.sleep(0.05)

            animated_results.append({
                "Run #": run_i,
                "Status": final_position.status,
                "PnL (₹)": round(final_position.pnl, 2),
            })

            if final_position.status == "CLOSED_TP":
                st.success(f"Run {run_i}: **CLOSED_TP** — ₹{final_position.pnl:,.2f}")
            elif final_position.status == "CLOSED_SL":
                st.error(f"Run {run_i}: **CLOSED_SL** — ₹{final_position.pnl:,.2f}")
            else:
                st.warning(f"Run {run_i}: **{final_position.status}** — ₹{final_position.pnl:,.2f}")

        # ----- Phase 2: Stream remaining runs into a live-updating table -----
        remaining = n_total - n_animate
        all_results = list(animated_results)

        if remaining > 0:
            st.markdown(f"### Phase 2 — Streaming {remaining} more runs")

            stats_ph = st.empty()
            table_ph = st.empty()

            for run_i in range(n_animate + 1, n_total + 1):
                broker = PaperBroker(risk_free_rate=0.06, lot_size=lot_size)
                risk = RiskManager(
                    profit_target_pct=tp_pct, stop_loss_pct=sl_pct, exit_dte=1.0,
                    tp_basis="max_profit" if tp_mode == "% of max profit" else "premium",
                    max_profit=abs(metrics["max_profit"]),
                )
                engine = SimulationEngine(broker=broker, risk=risk, steps=steps)
                final_position, _ = engine.run_with_history(
                    strategy=strategy, spot=spot, iv=iv, dte=dte,
                )

                all_results.append({
                    "Run #": run_i,
                    "Status": final_position.status,
                    "PnL (₹)": round(final_position.pnl, 2),
                })

                pnls = [r["PnL (₹)"] for r in all_results]
                statuses = [r["Status"] for r in all_results]
                wins = sum(1 for s in statuses if s == "CLOSED_TP")
                losses = sum(1 for s in statuses if s == "CLOSED_SL")
                timeouts = sum(1 for s in statuses if s == "CLOSED_TIME")
                avg = sum(pnls) / len(pnls)

                with stats_ph.container():
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Runs completed", f"{len(all_results)}/{n_total}")
                    s2.metric("Win rate", f"{wins/len(all_results)*100:.1f}%")
                    s3.metric("Avg PnL", f"₹{avg:,.2f}")
                    s4.metric("Total PnL", f"₹{sum(pnls):,.2f}")

                    s5, s6, s7, s8 = st.columns(4)
                    s5.metric("TP hits", wins)
                    s6.metric("SL hits", losses)
                    s7.metric("Time exits", timeouts)
                    s8.metric("Best / Worst",
                              f"₹{max(pnls):,.0f} / ₹{min(pnls):,.0f}")

                with table_ph.container():
                    st.dataframe(all_results, use_container_width=True,
                                 hide_index=True, height=400)

                time.sleep(0.02)

        # ----- Final summary -----
        st.divider()
        st.subheader(f"Final summary — {n_total} runs")
        pnls = [r["PnL (₹)"] for r in all_results]
        statuses = [r["Status"] for r in all_results]
        wins = sum(1 for s in statuses if s == "CLOSED_TP")
        losses = sum(1 for s in statuses if s == "CLOSED_SL")
        timeouts = sum(1 for s in statuses if s == "CLOSED_TIME")
        avg = sum(pnls) / n_total

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Win rate", f"{wins/n_total*100:.1f}%")
        f2.metric("Avg PnL", f"₹{avg:,.2f}")
        f3.metric("Total PnL", f"₹{sum(pnls):,.2f}")
        f4.metric("Best / Worst",
                  f"₹{max(pnls):,.0f} / ₹{min(pnls):,.0f}")

        f5, f6, f7, f8 = st.columns(4)
        f5.metric("TP hits", wins)
        f6.metric("SL hits", losses)
        f7.metric("Time exits", timeouts)
        std = (sum((p - avg)**2 for p in pnls) / n_total) ** 0.5
        f8.metric("Std dev", f"₹{std:,.2f}")

        fig = go.Figure(go.Histogram(
            x=pnls, nbinsx=30, marker_color="cyan",
        ))
        fig.add_vline(x=0, line=dict(color="white", dash="dash"))
        fig.update_layout(
            title="PnL distribution",
            xaxis_title="PnL (₹)", yaxis_title="Frequency",
            template="plotly_dark", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        if avg > 0 and wins / n_total > 0.5:
            st.success(f"**Positive expectancy.** Avg ₹{avg:,.2f}, "
                       f"win rate {wins/n_total*100:.1f}%.")
        else:
            st.error(f"**Negative or neutral expectancy.** Avg ₹{avg:,.2f}, "
                     f"win rate {wins/n_total*100:.1f}%.")


# ----- Tab 3: Monte Carlo -----
with tab_mc:
    st.write("Run the same strategy many times with different random paths.")
    n_runs = st.slider("Number of runs", 20, 500, 100, 20)
    run_mc = st.button("Run Monte Carlo", type="primary")

    if run_mc:
        strategy = build_strategy()
        metrics = compute_position_metrics(strategy, spot, iv, dte, lot_size)

        with st.spinner(f"Running {n_runs} simulations..."):
            result = run_monte_carlo(
                strategy=strategy, spot=spot, iv=iv, dte=dte,
                lot_size=lot_size, tp_pct=tp_pct, sl_pct=sl_pct,
                tp_basis="max_profit" if tp_mode == "% of max profit" else "premium",
                max_profit=abs(metrics["max_profit"]),
                n_runs=n_runs, steps=steps, seed=42,
            )

        st.subheader(f"Results — {n_runs} runs")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Win rate", f"{result.win_rate*100:.1f}%")
        m2.metric("Average PnL", f"₹{result.avg_pnl:,.2f}")
        m3.metric("Best", f"₹{result.best_pnl:,.2f}")
        m4.metric("Worst", f"₹{result.worst_pnl:,.2f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Median PnL", f"₹{result.median_pnl:,.2f}")
        m6.metric("Std dev", f"₹{result.std_pnl:,.2f}")
        counts = result.status_counts()
        m7.metric("TP hits", counts.get("CLOSED_TP", 0))
        m8.metric("SL hits", counts.get("CLOSED_SL", 0))

        st.subheader("Individual runs")
        runs_rows = []
        for i, (pnl, status) in enumerate(zip(result.pnls, result.statuses), start=1):
            runs_rows.append({
                "Run #": i,
                "Status": status,
                "PnL (₹)": round(pnl, 2),
            })
        st.dataframe(runs_rows, use_container_width=True, height=400)

        st.subheader("PnL distribution")
        fig = go.Figure(go.Histogram(
            x=result.pnls, nbinsx=30, marker_color="cyan",
        ))
        fig.add_vline(x=0, line=dict(color="white", dash="dash"))
        fig.update_layout(
            xaxis_title="PnL (₹)", yaxis_title="Frequency",
            template="plotly_dark", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        if result.avg_pnl > 0 and result.win_rate > 0.5:
            st.success(f"**Positive expectancy.** Average PnL ₹{result.avg_pnl:,.2f} "
                       f"over {n_runs} runs. Strategy has edge.")
        else:
            st.error(f"**Negative or neutral expectancy.** Average PnL "
                     f"₹{result.avg_pnl:,.2f} over {n_runs} runs. "
                     f"Adjust TP/SL or skip this strategy.")