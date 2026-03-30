import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simulation_core import MasterSimulation
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
import time

# ==========================================
# FIX: THE PICKLE BLUEPRINT
# ==========================================
class ContinuousCohortEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = BayesianGaussianMixture(
            n_components=3,
            weight_concentration_prior_type='dirichlet_process',
            warm_start=True,
            random_state=42
        )

st.set_page_config(page_title="AcademaSync | Spectator Mode", layout="wide")

# ==========================================
# 1. INITIALIZATION & HELPERS
# ==========================================
if 'sim' not in st.session_state:
    with st.spinner("Booting the Matrix... (Loading NLP Transformers takes ~30 seconds)"):
        st.session_state.sim = MasterSimulation(num_agents=200)
        print("[+] Matrix Online! Rendering UI...")
    st.session_state.selected_agent = 0

# ==========================================
# 1. INITIALIZATION & HELPERS
# ==========================================
if 'sim' not in st.session_state:
    with st.spinner("Booting the Matrix... (Loading NLP Transformers takes ~30 seconds)"):
        st.session_state.sim = MasterSimulation(num_agents=200)
        print("[+] Matrix Online! Rendering UI...")
    st.session_state.selected_agent = 0

sim = st.session_state.sim


def generate_tips(burnout, forecast):
    """Generates exactly 3 Pomodoro and 3 Wellness tips based on predictive telemetry."""
    tips = {"pomodoro": [], "wellness": []}

    # Pomodoro
    if forecast > 8.0:
        tips["pomodoro"].append("🎯 **DAILY TARGET:** Maximum **1.5 Hours** total. Engine 3 predicts crash if exceeded.")
        tips["pomodoro"].extend([
            "**Micro-Sprints:** Use 15m focus / 5m break to survive high cognitive load.",
            "**Task Slicing:** Break current assignment into 3 trivial sub-tasks.",
            "**Hard Stop:** Engine 3 predicts failure if you study past 9 PM tonight."
        ])
    elif burnout > 5.0:
        tips["pomodoro"].append("🎯 **DAILY TARGET:** Moderate **3.0 Hours** total. Prioritize consistency over volume.")
        tips["pomodoro"].extend([
            "**Standard Protocol:** Use 50m focus / 10m break.",
            "**Context Switching:** Avoid switching subjects; stick to one context.",
            "**Active Recall:** Test yourself at the end of each block."
        ])
    else:
        tips["pomodoro"].append("🎯 **DAILY TARGET:** Authorized for **5.0 - 6.0 Hours** total. Velocity is optimal.")
        tips["pomodoro"].extend([
            "**Deep Work Mode:** 90m uninterrupted flow state authorized.",
            "**Eat the Frog:** Tackle the highest difficulty task in Engine 4 first.",
            "**Momentum:** Chain tasks without breaking context."
        ])

    # Wellness
    if burnout > 7.5:
        tips["wellness"].extend([
            "**Screen Detachment:** Leave the room during your next break.",
            "**Sleep Debt:** You must secure 8+ hours tonight to lower tomorrow's forecast.",
            "**Social Reset:** Contact a friend. Isolation multiplies burnout."
        ])
    else:
        tips["wellness"].extend([
            "**Hydration:** Maintain water intake to sustain current cognitive flow.",
            "**Light Activity:** A 15-minute walk will improve afternoon retention.",
            "**Consistent Sleep:** Maintain your current sleep schedule; it is working."
        ])
    return tips


# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("⏳ Time Control")
    st.metric("Matrix Day", f"{sim.day}")

    c1, c2, c3 = st.columns(3)
    if c1.button("▶️ 1 Day"):
        sim.tick(skip_llm=False)
        st.rerun()
    if c2.button("⏭️ 3 Days"):
        with st.spinner("Simulating 72 hours of swarm telemetry..."):
            for i in range(3): sim.tick(skip_llm=(i != 2))
        st.rerun()
    if c3.button("⏩ 1 Week"):
        with st.spinner("Simulating 168 hours of swarm telemetry..."):
            for i in range(7): sim.tick(skip_llm=(i != 6))
        st.rerun()

    st.divider()
    st.subheader("Trending in 3'rd class Computer Science:")
    st.success(f"🔥 {sim.global_trend}")
    st.caption("Engine 1 Semantic Medoid. Applying +1.5x gravity to Engine 4 Recommenders globally.")

    st.divider()
    st.subheader("🤖 LLM Engine Log")
    st.code(sim.llm_log, language="log")

    # Top Ranking
    st.divider()
    st.subheader("🏆 Top Agents (Velocity)")
    top_agents = sorted(sim.agents, key=lambda a: a.xp, reverse=True)[:5]
    for i, a in enumerate(top_agents):
        risk = "🚨" if a.burnout > 8.0 else "🟢"
        st.markdown(f"**#{i + 1} Agent {a.id}** | {a.xp:,} XP | {risk}")

# ==========================================
# 3. UI TABS
# ==========================================
st.title("🌐 AcademaSync Master Simulation")
tab_macro, tab_micro = st.tabs(["🌌 The Swarm (Macro)", "🔍 Agent Inspector (Micro)"])

df_state = pd.DataFrame(
    [{"ID": a.id, "XP": a.xp, "Burnout": a.burnout, "Status": a.status, "Task": a.active_task} for a in sim.agents])

# ---------------------------------------------------------
# TAB 1: MACRO VIEW
# ---------------------------------------------------------
with tab_macro:
    c1, c2 = st.columns([3, 1])
    with c1:
        fig = px.scatter(
            df_state, x="XP", y="Burnout", color="Status",
            hover_data=["ID", "Task"], custom_data=["ID"], opacity=0.5, template="plotly_dark",
            color_discrete_map={"Normal 🟢": "#10B981", "Flow State ⚡": "#3B82F6", "Burned Out 📉": "#EF4444",
                                "Sick 🤒": "#F59E0B"}
        )
        # Lock-on targeting ring
        sel_agent = next(a for a in sim.agents if a.id == st.session_state.selected_agent)
        fig.add_scatter(x=[sel_agent.xp], y=[sel_agent.burnout], mode="markers",
                        marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(color="white", width=3)),
                        name=f"Target: Agent #{sel_agent.id}", hoverinfo="skip")
        fig.add_hline(y=7.5, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        fig.update_layout(height=600)

        # --- NEW: CLICKABLE PLOT ---
        selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

        if selection and selection.selection.points:
            try:
                # Extract the ID from the clicked data point
                clicked_id = int(selection.selection.points[0].customdata[0])
                if st.session_state.selected_agent != clicked_id:
                    st.session_state.selected_agent = clicked_id
                    st.rerun()
            except Exception:
                pass

    with c2:
        st.divider()
        st.metric("Critical (>7.5 ρ)", len(df_state[df_state['Burnout'] > 7.5]))
        st.bar_chart(df_state['Status'].value_counts())

# ---------------------------------------------------------
# TAB 2: MICRO VIEW (THE SHOWCASE)
# ---------------------------------------------------------
# ---------------------------------------------------------
# TAB 2: MICRO VIEW (THE SHOWCASE)
# ---------------------------------------------------------
with tab_micro:
    # --- NEW: MOVED DROPDOWN ---
    st.session_state.selected_agent = st.selectbox("🔍 Manual Override (Or click student on Macro Plot):",
                                                   df_state['ID'].tolist(),
                                                   index=df_state['ID'].tolist().index(st.session_state.selected_agent))

    agent = next(a for a in sim.agents if a.id == st.session_state.selected_agent)

    # 1. Top Level Daily Stats
    st.markdown(f"## 👤 Agent #{agent.id} Dashboard")
    st.markdown(f"**Current Status:** {agent.status}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Velocity", f"{agent.xp:,} XP")
    m2.metric("Daily Focus Time", f"{agent.daily_focus_h:.1f}h")
    m3.metric("Tasks Done Today", f"{agent.tasks_done_today} / {agent.tasks_total_today}")

    # Engine 3 Forecast metric
    forecast_shift = agent.forecasted_burnout - agent.burnout
    shift_color = "inverse" if forecast_shift > 0 else "normal"
    m4.metric("Engine 3 Forecast", f"{agent.forecasted_burnout:.2f} ρ",
              delta=f"{forecast_shift:+.2f} shift", delta_color=shift_color)

    st.divider()

    # 2. Comprehensive Telemetry
    col_att, col_study, col_task = st.columns(3)

    with col_att:
        st.subheader("📅 Attendance Tracker")
        att_rate = agent.classes_attended / agent.total_classes
        st.markdown(f"**Overall Rate:** `{att_rate * 100:.1f}%`")

        fig_att = go.Figure(data=[go.Pie(
            labels=['Present', 'Late', 'Skipped (Micro)', 'Cancelled (Macro)'],
            values=[agent.classes_attended, agent.classes_late, agent.classes_skipped, agent.classes_cancelled],
            hole=.6,
            marker_colors=['#10B981', '#F59E0B', '#EF4444', '#6B7280']
        )])
        fig_att.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_att, use_container_width=True)
        st.caption(f"Skipped: {agent.classes_skipped} | Cancelled: {agent.classes_cancelled}")

    with col_study:
        st.subheader("⏱️ Study Statistics")
        st.markdown(f"- **Total Focus Sessions:** `{agent.focus_sessions}`")
        st.markdown(f"- **Average Session:** `{agent.avg_session_min} min`")
        st.markdown(f"- **Current Streak:** `{agent.current_streak} days`")
        st.markdown(f"- **Longest Streak:** `{agent.longest_streak} days`")
        st.markdown(f"- **Total Classes:** `{agent.total_classes}`")

    with col_task:
        st.subheader("🧠 Engine 2: NLP Analysis")
        st.markdown(f"> *\"{agent.todays_journal}\"*")
        delta = agent.history['nlp_delta'][-1] if agent.history['day'] else 0
        delta_color = "red" if delta > 0 else "green"
        st.markdown(f"**Burnout Delta:** :{delta_color}[**{delta:+.2f}**]")

    st.divider()

    # 3. Dynamic Tips & Engine 3
    col_tips, col_e3 = st.columns([1, 1])
    with col_tips:
        st.subheader("🤖 Predictive Advisories")
        tips = generate_tips(agent.burnout, agent.forecasted_burnout)
        st.markdown("#### 🍅 Pomodoro Guidelines")
        for t in tips["pomodoro"]: st.info(t)
        st.markdown("#### 💙 Wellness Directives")
        for t in tips["wellness"]: st.success(t)

    with col_e3:
        st.subheader("📈 Time-Series Trajectory")
        if agent.history['day']:
            df_hist = pd.DataFrame(agent.history)

            # --- NEW: ACTUAL VS PREDICTED CANDLESTICK ---
            # Open = Predicted Burnout (Yesterday's Burnout + Today's NLP Delta)
            # Close = Actual Burnout Today
            df_hist['Predicted'] = df_hist['burnout'].shift(1).fillna(df_hist['burnout']) + df_hist['nlp_delta']
            df_hist['High'] = df_hist[['Predicted', 'burnout']].max(axis=1) + 0.1
            df_hist['Low'] = df_hist[['Predicted', 'burnout']].min(axis=1) - 0.1

            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_hist['day'],
                open=df_hist['Predicted'],
                high=df_hist['High'],
                low=df_hist['Low'],
                close=df_hist['burnout'],
                name="Pred. Var.",
                increasing_line_color='#10B981',  # Red: Actual > Predicted (Bad)
                decreasing_line_color='#EF4444'  # Green: Actual < Predicted (Good Recovery)
            )])

            # Overlay the solid Actual Trend Line
            fig_candle.add_trace(go.Scatter(
                x=df_hist['day'], y=df_hist['burnout'],
                mode='lines+markers', line=dict(color='#3B82F6', width=2),
                name='Actual (ρ)'
            ))

            fig_candle.add_hline(y=7.5, line_dash="dash", line_color="red")
            fig_candle.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0),
                                     template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_candle, use_container_width=True)
        else:
            st.warning("Advance time to generate temporal history.")

    st.divider()

    # 4. Engine 4 Task Queue (Scrollable)
    st.subheader("🎯 Engine 4: Triage Queue")
    st.caption(
        f"Sorted by XGBRanker Pairwise Loss. Semantic Gravity applied for alignment with global trend: '{sim.global_trend}'")

    st.dataframe(
        agent.backlog[['Task_ID', 'Due_H', 'Diff', 'Base_LTR', 'Final_Score']].style.highlight_max(
            subset=['Final_Score'], color='rgba(59, 130, 246, 0.4)'),
        use_container_width=True,
        height=300,
        hide_index=True
    )