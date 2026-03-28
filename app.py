import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import math
import faiss
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. OS CONFIGURATION
# ==========================================
st.set_page_config(page_title="Engine 4 Strict Lab", page_icon="🎯", layout="wide")


# ==========================================
# 2. STRICT ML ENGINE LOADING
# ==========================================
@st.cache_resource
def load_ml_models():
    """Loads the ACTUAL serialized models. No fake math fallbacks."""
    engines = {"recommender": None, "embedder": None}

    # 1. Load the Pickled XGBRanker
    model_path = "output/recommender_engine.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            engines["recommender"] = pickle.load(f)
    else:
        st.error(f"❌ FATAL ERROR: {model_path} not found. You must run the Recommender Notebook first.")
        st.stop()

    # 2. Load the Live Semantic Encoder
    with st.spinner("Loading transformer weights into memory..."):
        engines["embedder"] = SentenceTransformer('all-MiniLM-L6-v2')

    return engines


engines = load_ml_models()

# ==========================================
# 3. STATE MANAGEMENT
# ==========================================
if 'task_matrix' not in st.session_state:
    st.session_state.task_matrix = pd.DataFrame([
        {"Task_ID": "Physics Lab Report", "Hours_to_Due": 12.0, "Difficulty": 4, "Est_Mins": 90.0},
        {"Task_ID": "Data Structures Assignment", "Hours_to_Due": 48.0, "Difficulty": 5, "Est_Mins": 120.0},
        {"Task_ID": "Read History Chapter 4", "Hours_to_Due": 72.0, "Difficulty": 2, "Est_Mins": 45.0},
        {"Task_ID": "Buy Groceries", "Hours_to_Due": 24.0, "Difficulty": 1, "Est_Mins": 40.0}
    ])


# ==========================================
# 4. REAL PIPELINE LOGIC
# ==========================================
def execute_semantic_search(task_names, cohort_topic):
    """Executes REAL FAISS vector density search."""
    if not cohort_topic.strip():
        return np.zeros(len(task_names))

    # 1. Vectorize Cohort Trend
    trend_embed = engines["embedder"].encode([cohort_topic])
    faiss.normalize_L2(trend_embed)

    # 2. Vectorize User Backlog
    task_embeds = engines["embedder"].encode(task_names)
    faiss.normalize_L2(task_embeds)

    # 3. Cosine Similarity (Dot Product of normalized vectors)
    similarities = np.dot(task_embeds, trend_embed.T).flatten()

    # Zero out negative correlations to prevent penalizing unrelated tasks
    return np.maximum(similarities, 0)


def generate_stat_driven_insight(row, burnout_val, time_label):
    """Generates UX strings strictly based on the variables."""
    insights = []

    # Burnout vs Difficulty check
    if burnout_val >= 7.0 and row['Difficulty'] <= 2:
        insights.append(
            f"Because your Burnout Index is critically high ({burnout_val}/10), the engine prioritized this low-friction task to protect your mental load.")
    elif burnout_val >= 7.0 and row['Hours_to_Due'] <= 24:
        insights.append(
            f"⚠️ Even though you are highly fatigued, the extreme temporal urgency ({row['Hours_to_Due']}h) overrode the safety protocols.")

    # Chronotype check
    if time_label == "Morning (High Energy)" and row['Difficulty'] >= 4:
        insights.append("Scheduled to align with your morning peak energy window for deep work.")
    elif time_label == "Afternoon (Slump)" and row['Difficulty'] <= 2:
        insights.append("Matched to your afternoon circadian dip (low complexity).")

    # Semantic check
    if row['Semantic_Boost'] > 0.4:
        insights.append(f"This concept strongly aligns with the current cohort semantic trend.")

    if not insights:
        insights.append("Balanced mathematically against your deadline and historical velocity.")

    return " ".join(insights)


def generate_tips(burnout_val):
    """Generates Pomodoro/Wellness advice strictly based on the telemetry state."""
    if burnout_val >= 7.5:
        return [
            "🚨 **CRITICAL OVERLOAD:** Limit execution to 25-minute micro-sprints.",
            "💤 **RECOVERY MANDATORY:** You must schedule a hard stop 2 hours before bed.",
            "🚶 **DETACHMENT:** Physical separation from your desk during breaks is required."
        ]
    elif burnout_val >= 5.0:
        return [
            "🔋 **MODERATE FATIGUE:** Utilize standard 50/10 Pomodoro blocks.",
            "💧 **HYDRATION:** Ensure consistent water intake to maintain focus.",
        ]
    else:
        return [
            "⚡ **PEAK VELOCITY:** You are cleared for 90-minute Deep Work cycles.",
            "🧠 **EAT THE FROG:** Tackle your highest complexity tasks immediately."
        ]


# ==========================================
# 5. UI LAYOUT & EXECUTION
# ==========================================
st.title("🎯 Strict Engine 4 Laboratory")
st.markdown("This lab routes inputs directly into your pickled `XGBRanker` and `all-MiniLM-L6-v2` models.")

with st.sidebar:
    st.header("🎛️ Algorithm State Controls")

    st.subheader("Engine 3 Telemetry (Burnout)")
    sim_burnout = st.slider("Current Burnout Index (ρ)", 1.0, 10.0, 3.0, 0.5)

    st.subheader("Time Dimension")
    sim_time = st.selectbox("Time of Day", ["Morning (High Energy)", "Afternoon (Slump)", "Evening (Wind Down)"])

    # Calculate Chronotype math based on selection
    if "Morning" in sim_time:
        hour_sin, hour_cos = 1.0, 0.0
    elif "Afternoon" in sim_time:
        hour_sin, hour_cos = 0.0, -1.0
    else:
        hour_sin, hour_cos = -1.0, 0.0

    st.subheader("Engine 1 Context")
    sim_cohort_trend = st.text_input("Trending Cohort Topic", value="Data Structures")
    sim_semantic_weight = st.slider("Semantic Gravity Weight", 0.0, 5.0, 1.5, 0.5)

col_main, col_add = st.columns([2, 1])

with col_add:
    st.subheader("📥 Matrix Injection")
    with st.form("task_injection"):
        new_task = st.text_input("Task Name")
        new_due = st.number_input("Hours Due", 1.0, 500.0, 24.0)
        new_diff = st.slider("Difficulty (ψ)", 1, 5, 3)
        new_est = st.number_input("Historical Mins (v)", 10, 600, 60)

        if st.form_submit_button("Inject", use_container_width=True):
            if new_task:
                new_row = {"Task_ID": new_task, "Hours_to_Due": new_due, "Difficulty": new_diff, "Est_Mins": new_est}
                st.session_state.task_matrix = pd.concat([st.session_state.task_matrix, pd.DataFrame([new_row])],
                                                         ignore_index=True)
                st.rerun()

with col_main:
    st.subheader("📊 Model Inference Output")

    df = st.session_state.task_matrix.copy()

    if not df.empty:
        # --- PHASE A: Construct 7D Feature Tensor for XGBRanker ---
        # Must exactly match the features list used during training
        X_infer = pd.DataFrame({
            'time_to_due_date': df['Hours_to_Due'],
            'dynamic_difficulty': df['Difficulty'],
            'user_burnout_index': [sim_burnout] * len(df),
            'historical_velocity': df['Est_Mins'],
            'hour_sin': [hour_sin] * len(df),
            'hour_cos': [hour_cos] * len(df),
            'temporal_velocity': df['Est_Mins'] * (0.9 if hour_sin > 0 else 1.1)
        })

        # --- PHASE B: Execute the Serialized Model ---
        # WE USE THE REAL PREDICT FUNCTION HERE
        base_utilities = engines["recommender"].predict(X_infer)

        # --- PHASE C: Execute the Real Semantic Engine ---
        semantic_boosts = execute_semantic_search(df['Task_ID'].tolist(), sim_cohort_trend)

        # --- PHASE D: The Unified Equation ---
        df['Base_LTR'] = base_utilities
        df['Semantic_Boost'] = semantic_boosts
        df['Final_Utility'] = df['Base_LTR'] + (df['Semantic_Boost'] * sim_semantic_weight)

        df = df.sort_values(by='Final_Utility', ascending=False).reset_index(drop=True)

        # --- RENDER ---
        for idx, row in df.iterrows():
            rank = idx + 1
            is_top = rank == 1

            with st.container(border=True):
                c_rank, c_info = st.columns([1, 6])

                with c_rank:
                    color = "#3B82F6" if is_top else "gray"
                    st.markdown(f"<h2 style='color:{color}; text-align:center;'>#{rank}</h2>", unsafe_allow_html=True)

                with c_info:
                    icon = "🔥" if row['Semantic_Boost'] > 0.4 else "📄"
                    st.markdown(f"#### {icon} {row['Task_ID']}")
                    st.caption(
                        f"**Due:** {row['Hours_to_Due']}h | **Diff:** {row['Difficulty']}/5 | **Base LTR:** {row['Base_LTR']:.2f} | **Semantic:** +{row['Semantic_Boost'] * sim_semantic_weight:.2f}")

                    if is_top:
                        insight_text = generate_stat_driven_insight(row, sim_burnout, sim_time)
                        st.success(f"💡 **Dynamic Insight:** {insight_text}")
    else:
        st.info("Matrix empty. Inject a task.")

st.divider()
st.subheader("🩺 Telemetry-Driven Advisories")
tips = generate_tips(sim_burnout)
for tip in tips:
    st.info(tip)