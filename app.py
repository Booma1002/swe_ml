import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import math
import plotly.express as px
import plotly.graph_objects as go
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import faiss

# ==========================================
# 1. OS CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="AcademaSync Master OS", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .metric-card { background: #1E293B; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
    .tip-box { background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3B82F6; padding: 10px; margin-bottom: 10px; border-radius: 4px;}
    .wellness-box { background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981; padding: 10px; margin-bottom: 10px; border-radius: 4px;}
    .alert-box { background: rgba(239, 68, 68, 0.1); border-left: 4px solid #EF4444; padding: 10px; margin-bottom: 10px; border-radius: 4px;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. UNIFIED MODEL LOADING (Strict)
# ==========================================
@st.cache_resource
def load_all_engines():
    engines = {"recommender": None, "sentiment": None, "embedder": None}

    try:
        with open("output/recommender_engine.pkl", "rb") as f:
            engines["recommender"] = pickle.load(f)
        with open("output/sentiment_model.pkl", "rb") as f:
            engines["sentiment"] = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Missing Pickles: {e}. Please ensure Engine 2 and 4 are compiled.")
        st.stop()

    with st.spinner("Loading Transformer Weights..."):
        engines["embedder"] = SentenceTransformer('all-MiniLM-L6-v2')

    return engines


engines = load_all_engines()


# ==========================================
# 3. ENGINE 1: ADVANCED SEMANTIC SWARM
# ==========================================
class SemanticCohortEngine:
    def __init__(self, embedder):
        self.scaler = StandardScaler()
        self.model = BayesianGaussianMixture(n_components=3, weight_concentration_prior_type='dirichlet_process',
                                             random_state=42)
        self.feature_names = ['weekly_xp_velocity', 'attendance_rate', 'average_burnout_index']
        self.embedder = embedder
        self.state_memory = pd.DataFrame()

    def process_batch(self, df_batch):
        X_raw = df_batch[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X_raw)
        self.model.fit(X_scaled)

        df_batch['cluster_label'] = self.model.predict(X_scaled)
        self.state_memory = df_batch
        return True

    def extract_semantic_trend(self):
        if self.state_memory.empty: return "Analyzing..."

        raw_centroids = self.scaler.inverse_transform(self.model.means_)
        hp_cluster_label = np.argmax(raw_centroids[:, 0])

        hp_tasks = self.state_memory[self.state_memory['cluster_label'] == hp_cluster_label][
            'active_task'].dropna().unique()
        if len(hp_tasks) == 0: return "Unknown"

        embeddings = self.embedder.encode(hp_tasks)
        mean_embedding = np.mean(embeddings, axis=0).reshape(1, -1)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        _, I = index.search(mean_embedding, 1)

        return hp_tasks[I[0][0]]


@st.cache_data
def get_global_data():
    np.random.seed(42)
    size = 2000
    df = pd.DataFrame({
        'weekly_xp_velocity': np.random.normal(6000, 2000, size),
        'attendance_rate': np.random.uniform(0.5, 1.0, size),
        'average_burnout_index': np.random.normal(5.0, 2.0, size),
        'active_task': np.random.choice([
            "phys hw", "Physics Homework", "physics assignment",
            "C++ pointers", "Data structures array", "Jade Library architecture",
            "Machine Learning backprop", "Calculus limits"
        ], size=size)
    })
    # Clean up negatives from normal distribution
    df['weekly_xp_velocity'] = np.clip(df['weekly_xp_velocity'], 0, 15000)
    df['average_burnout_index'] = np.clip(df['average_burnout_index'], 1.0, 10.0)
    return df


if 'cohort_engine' not in st.session_state:
    st.session_state.cohort_engine = SemanticCohortEngine(engines['embedder'])
    st.session_state.cohort_engine.process_batch(get_global_data())

if 'user_state' not in st.session_state:
    st.session_state.user_state = {"burnout": 4.0, "xp": 4500, "sleep": 7.0}

if 'task_matrix' not in st.session_state:
    st.session_state.task_matrix = pd.DataFrame([
        {"Task_ID": "Physics Quiz Prep", "Hours_to_Due": 12.0, "Difficulty": 4, "Est_Mins": 45.0},
        {"Task_ID": "Implement C++ BST", "Hours_to_Due": 48.0, "Difficulty": 5, "Est_Mins": 120.0},
        {"Task_ID": "Read Literature", "Hours_to_Due": 72.0, "Difficulty": 2, "Est_Mins": 60.0}
    ])


def generate_contextual_tips(burnout, sleep, time_of_day):
    tips = {"pomodoro": [], "wellness": [], "ai_recs": []}
    if burnout >= 7.0:
        tips["pomodoro"].append("⏱️ **Timer: 25m Focus / 10m Break.** (Fatigue override active).")
        tips["wellness"].append(
            "🚨 **CRITICAL:** High cognitive load detected. Strict physical detachment required during breaks.")
    elif burnout >= 4.0:
        tips["pomodoro"].append("⏱️ **Timer: 50m Focus / 10m Break.** Standard operating procedure.")
        tips["wellness"].append("💧 **Hydration:** Keep water nearby to sustain medium-level focus.")
    else:
        tips["pomodoro"].append("⏱️ **Timer: 90m Deep Work.** You are in peak condition.")
        tips["wellness"].append("⚡ **Momentum:** Maximize your flow state, but don't skip meals.")

    if sleep < 6.0:
        tips["wellness"].append(f"💤 **Sleep Debt:** {sleep}h logged. Aim for 8 hours tonight for memory consolidation.")

    if "Morning" in time_of_day:
        tips["ai_recs"].append("🌅 You perform best in the morning - schedule difficult tasks immediately.")
    elif "Afternoon" in time_of_day:
        tips["ai_recs"].append("☕ Mid-day energy dip approaching. Clear administrative tasks.")
    return tips


# ==========================================
# 6. MASTER UI ROUTING
# ==========================================
with st.sidebar:
    st.header("🔑 Access Control")
    current_role = st.radio("System View", ["🎓 Student (Triage)", "👨‍🏫 Staff (Telemetry)"])
    st.divider()

if current_role == "🎓 Student (Triage)":
    # ---------------------------------------------------------
    # STUDENT VIEW
    # ---------------------------------------------------------
    st.title("⚡ AcademaSync Master OS")

    with st.sidebar:
        st.header("🧠 Daily Ingestion")
        time_sim = st.selectbox("Current Time", ["Morning (High Energy)", "Afternoon (Slump)", "Evening (Wind Down)"])
        st.divider()
        st.subheader("1. Physiological Telemetry")
        mood_map = {"😄 Great": -1.0, "😊 Good": -0.5, "😐 Okay": 0.0, "😞 Bad": +1.0}
        mood = st.radio("Mood", list(mood_map.keys()), index=2, horizontal=True)
        sleep = st.slider("Sleep (Hours)", 0.0, 12.0, st.session_state.user_state['sleep'], 0.5)
        stress = st.select_slider("Stress Load", ["Low", "Medium", "High"], "Medium")
        st.divider()
        st.subheader("2. NLP Journal (Engine 2)")
        journal = st.text_area("Reflection", placeholder="How did today go?", height=150)

        if st.button("Save & Execute Inference", use_container_width=True, type="primary"):
            base_delta = mood_map[mood] + (1.0 if stress == "High" else -0.5 if stress == "Low" else 0)
            nlp_delta = 0
            if journal.strip():
                probs = engines["sentiment"].predict_proba([journal])[0]
                nlp_delta = (probs[0] - probs[1]) * 1.5

            total_delta = base_delta + nlp_delta
            st.session_state.user_state['burnout'] = np.clip(st.session_state.user_state['burnout'] + total_delta, 1.0,
                                                             10.0)
            st.session_state.user_state['sleep'] = sleep
            st.success(f"Log Saved! Burnout adjusted by {total_delta:+.2f} points.")
            st.rerun()

        st.divider()
        st.metric("Current Burnout Index (ρ)", f"{st.session_state.user_state['burnout']:.1f} / 10")

    col_recs, col_tips = st.columns([2, 1])
    live_trend = st.session_state.cohort_engine.extract_semantic_trend()

    with col_tips:
        st.subheader("💡 Dynamic Advisory")
        tips = generate_contextual_tips(st.session_state.user_state['burnout'], st.session_state.user_state['sleep'],
                                        time_sim)
        for t in tips["pomodoro"]: st.markdown(f"<div class='tip-box'>{t}</div>", unsafe_allow_html=True)
        for t in tips["wellness"]: st.markdown(f"<div class='wellness-box'>{t}</div>", unsafe_allow_html=True)

    with col_recs:
        st.subheader("🤖 Engine Insights")
        if st.session_state.user_state['burnout'] >= 7.0:
            st.markdown(
                "<div class='alert-box'>⚠️ <b>AUTO-BURNOUT ALERT:</b> System is actively penalizing high-difficulty tasks to protect cognitive limits.</div>",
                unsafe_allow_html=True)
        for t in tips["ai_recs"]: st.info(t)
        st.warning(
            f"🔥 **Cohort Gravity:** Semantic analysis detects dense activity around **'{live_trend}'**. Recommender is applying +1.5x alignment boost.")

    st.divider()
    st.subheader("🎯 Active Matrix (Engine 4)")

    df_tasks = st.session_state.task_matrix.copy()

    if not df_tasks.empty:
        hour_sin, hour_cos = (1, 0) if "Morning" in time_sim else (0, -1) if "Afternoon" in time_sim else (-1, 0)

        X_infer = pd.DataFrame({
            'time_to_due_date': df_tasks['Hours_to_Due'],
            'dynamic_difficulty': df_tasks['Difficulty'],
            'user_burnout_index': [st.session_state.user_state['burnout']] * len(df_tasks),
            'historical_velocity': df_tasks['Est_Mins'],
            'hour_sin': [hour_sin] * len(df_tasks),
            'hour_cos': [hour_cos] * len(df_tasks),
            'temporal_velocity': df_tasks['Est_Mins']
        })

        df_tasks['Base_LTR'] = engines["recommender"].predict(X_infer)
        trend_vec = engines["embedder"].encode([live_trend])
        faiss.normalize_L2(trend_vec)
        task_vecs = engines["embedder"].encode(df_tasks['Task_ID'].tolist())
        faiss.normalize_L2(task_vecs)
        similarities = np.dot(task_vecs, trend_vec.T).flatten()
        df_tasks['Semantic_Boost'] = np.maximum(similarities, 0)
        df_tasks['Final_Utility'] = df_tasks['Base_LTR'] + (df_tasks['Semantic_Boost'] * 1.5)
        df_tasks = df_tasks.sort_values(by='Final_Utility', ascending=False).reset_index(drop=True)

        c1, c2, c3, c4 = st.columns([1, 4, 2, 2])
        c1.markdown("**Rank**");
        c2.markdown("**Task**");
        c3.markdown("**Utility ($\mu$)**");
        c4.markdown("**Status**")

        for idx, row in df_tasks.iterrows():
            rank = idx + 1
            is_top = rank == 1
            color = "#3B82F6" if is_top else "white"
            c1, c2, c3, c4 = st.columns([1, 4, 2, 2])
            c1.markdown(f"<h3 style='color:{color}'>#{rank}</h3>", unsafe_allow_html=True)
            c2.markdown(
                f"**{row['Task_ID']}**<br><small>Due: {row['Hours_to_Due']}h | Diff: {row['Difficulty']}/5</small>",
                unsafe_allow_html=True)
            c3.markdown(
                f"**{row['Final_Utility']:.2f}**<br><small>Base: {row['Base_LTR']:.1f} | Sem: +{row['Semantic_Boost'] * 1.5:.1f}</small>",
                unsafe_allow_html=True)
            with c4:
                st.button("Complete", key=f"btn_{idx}", use_container_width=True,
                          type="primary" if is_top else "secondary")

else:
    # ---------------------------------------------------------
    # STAFF / ADMIN VIEW (Anonymized)
    # ---------------------------------------------------------
    st.title("👨‍🏫 Institutional Telemetry (Anonymized)")
    st.markdown("Aggregated stress and velocity metrics across the student body. No PII is exposed.")

    df_swarm = st.session_state.cohort_engine.state_memory

    if not df_swarm.empty:
        # Map unstructured strings into loose subject buckets for cleaner admin reporting
        def categorize_task(task):
            t = task.lower()
            if "phys" in t: return "Physics"
            if "c++" in t or "data structure" in t or "jade" in t: return "C++ / Systems"
            if "machine" in t or "backprop" in t: return "Machine Learning"
            if "calc" in t: return "Mathematics"
            return "Other"


        df_swarm['Subject_Category'] = df_swarm['active_task'].apply(categorize_task)

        # Top Level Metrics
        m1, m2, m3 = st.columns(3)
        global_burnout = df_swarm['average_burnout_index'].mean()
        critical_count = len(df_swarm[df_swarm['average_burnout_index'] >= 7.5])

        m1.metric("Total Active Students", f"{len(df_swarm):,}")
        m2.metric("Global Avg Burnout", f"{global_burnout:.2f} / 10",
                  delta="Warning" if global_burnout > 6 else "Stable", delta_color="inverse")
        m3.metric("Students at Critical Risk (>7.5 ρ)", f"{critical_count:,}",
                  delta=f"{(critical_count / len(df_swarm)) * 100:.1f}% of body", delta_color="inverse")

        st.divider()

        # Aggregated Departmental Data
        st.subheader("Departmental Stress Distribution")

        dept_stats = df_swarm.groupby('Subject_Category').agg(
            Active_Students=('Subject_Category', 'count'),
            Avg_Burnout=('average_burnout_index', 'mean'),
            Avg_XP=('weekly_xp_velocity', 'mean')
        ).reset_index()

        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            fig_bar = px.bar(dept_stats, x='Subject_Category', y='Avg_Burnout', color='Avg_Burnout',
                             title="Average Burnout (ρ) by Subject Area",
                             color_continuous_scale="Reds", range_y=[1, 10],
                             labels={"Subject_Category": "Subject Area", "Avg_Burnout": "Mean Burnout Index"})
            fig_bar.add_hline(y=7.0, line_dash="dash", line_color="red", annotation_text="Danger Zone")
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_chart2:
            fig_scatter = px.scatter(dept_stats, x='Avg_XP', y='Avg_Burnout', size='Active_Students',
                                     color='Subject_Category',
                                     title="Academic Velocity vs. Fatigue",
                                     labels={"Avg_XP": "Mean Weekly XP Velocity", "Avg_Burnout": "Mean Burnout Index"})
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()
        st.subheader("🌌 Engine 1: Semantic Manifold (Global Cluster)")
        st.caption("Visualizing the unsupervised topological clustering of the entire student body.")
        fig_swarm = px.scatter(df_swarm, x="weekly_xp_velocity", y="average_burnout_index", color="cluster_label",
                               hover_data=["active_task"], opacity=0.3, template="plotly_dark",
                               title="Global Cohort Distribution (BGM Latent Space)")
        st.plotly_chart(fig_swarm, use_container_width=True)

    else:
        st.info("No telemetry data available from Engine 1.")