import os
import math
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import json
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler


# ==========================================
# INTERNAL STATE MEMORY
# ==========================================
STATE_FILE = "output/engine_state.json"

def get_current_trend():
    """Reads the trend internally so Node.js doesn't have to send it."""
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            return state.get("global_trend", "Core Curriculum Focus")
    except (FileNotFoundError, json.JSONDecodeError):
        return "Core Curriculum Focus"

def save_current_trend(trend_name):
    """Saves the trend internally when Engine 1 runs."""
    # Ensure the directory exists
    os.makedirs("output", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"global_trend": trend_name}, f)

# ==========================================
# 1. INITIALIZATION & MODEL LOADING
# ==========================================
app = FastAPI(
    title="AcademaSync Inference & Training Engine",
    description="Stateless ML Microservice for Node.js Backend Integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[*] Booting AcademaSync Neural Cores...")

try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"[!] Warning: Could not load SentenceTransformer: {e}")
    embedder = None

# --- PICKLE BLUEPRINT ---
# Python needs this class definition to know how to unpack Engine 1
class ContinuousCohortEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = BayesianGaussianMixture(
            n_components=3,
            weight_concentration_prior_type='dirichlet_process',
            warm_start=True,
            random_state=42
        )

import __main__
__main__.ContinuousCohortEngine = ContinuousCohortEngine

def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"[!] Warning: Missing model at {path}")
    return None

engine_1_cohort = load_pickle("output/continuous_cohort_model.pkl")
engine_2_nlp = load_pickle("output/sentiment_model.pkl")
engine_3_ts = load_pickle("output/time_series_model.pkl")
engine_4_rec = load_pickle("output/recommender_engine.pkl")

print("[+] Systems Online.")

# ==========================================
# 2. JSON CONTRACTS (PYDANTIC SCHEMAS)
# ==========================================

# --- Engine 1 ---
class StudentTelemetry(BaseModel):
    weekly_xp_velocity: float
    attendance_rate: float
    on_time_completion_rate: float
    average_burnout_index: float

class BatchTelemetryRequest(BaseModel):
    data: List[StudentTelemetry]

# --- Engine 2 ---
class SentimentRequest(BaseModel):
    journal_text: str

class SentimentTrainingRow(BaseModel):
    text: str
    mood_label: int  # 0 for negative, 1 for positive

class SentimentRetrainPayload(BaseModel):
    training_data: List[SentimentTrainingRow]

# --- Engine 3 ---
class TimeSeriesRequest(BaseModel):
    current_burnout: float
    sleep_hours_lag_1: Optional[float] = None
    sleep_hours_lag_2: Optional[float] = None
    sleep_hours_lag_3: Optional[float] = None
    pomodoro_hours_lag_1: Optional[float] = None
    pomodoro_hours_lag_2: Optional[float] = None
    pomodoro_hours_lag_3: Optional[float] = None
    tasks_in_queue_lag_1: Optional[int] = None
    tasks_in_queue_lag_2: Optional[int] = None
    tasks_in_queue_lag_3: Optional[int] = None
    days_to_exam_lag_1: Optional[int] = None
    days_to_exam_lag_2: Optional[int] = None
    days_to_exam_lag_3: Optional[int] = None
    burnout_index_lag_1: Optional[float] = None
    burnout_index_lag_2: Optional[float] = None
    burnout_index_lag_3: Optional[float] = None

# --- Engine 4 ---
class TaskItem(BaseModel):
    task_id: str
    title: str
    due_h: float
    difficulty: int
    historical_velocity: float = 60.0
    temporal_velocity: float = 60.0

class RecommenderRequest(BaseModel):
    current_hour: int = Field(..., description="0-23 integer for chronotype math")
    user_burnout_index: float
    tasks: List[TaskItem]

# ==========================================
# 3. INFERENCE ENDPOINTS (DAILY USAGE)
# ==========================================

@app.post("/api/v1/infer/sentiment")
def infer_sentiment(payload: SentimentRequest):
    if not engine_2_nlp:
        raise HTTPException(status_code=503, detail="Sentiment model offline.")

    probs = engine_2_nlp.predict_proba([payload.journal_text])[0]
    delta = (probs[0] - probs[1]) * 1.5
    return {"burnout_delta": round(delta, 3)}

@app.post("/api/v1/infer/forecast")
def infer_forecast(payload: TimeSeriesRequest):
    if not engine_3_ts:
        raise HTTPException(status_code=503, detail="Time-Series model offline.")

    if payload.sleep_hours_lag_3 is None:
        return {
            "forecasted_burnout": payload.current_burnout,
            "status": "cold_start_fallback_active"
        }

    X_temporal = pd.DataFrame([{
        'sleep_hours_lag_1': payload.sleep_hours_lag_1,
        'sleep_hours_lag_2': payload.sleep_hours_lag_2,
        'sleep_hours_lag_3': payload.sleep_hours_lag_3,
        'pomodoro_hours_lag_1': payload.pomodoro_hours_lag_1,
        'pomodoro_hours_lag_2': payload.pomodoro_hours_lag_2,
        'pomodoro_hours_lag_3': payload.pomodoro_hours_lag_3,
        'tasks_in_queue_lag_1': payload.tasks_in_queue_lag_1,
        'tasks_in_queue_lag_2': payload.tasks_in_queue_lag_2,
        'tasks_in_queue_lag_3': payload.tasks_in_queue_lag_3,
        'days_to_exam_lag_1': payload.days_to_exam_lag_1,
        'days_to_exam_lag_2': payload.days_to_exam_lag_2,
        'days_to_exam_lag_3': payload.days_to_exam_lag_3,
        'burnout_index_lag_1': payload.burnout_index_lag_1,
        'burnout_index_lag_2': payload.burnout_index_lag_2,
        'burnout_index_lag_3': payload.burnout_index_lag_3
    }])

    forecast = float(np.clip(engine_3_ts.predict(X_temporal)[0], 1.0, 10.0))
    return {"forecasted_burnout": round(forecast, 2), "status": "success"}


@app.post("/api/v1/infer/cohort_sync")
def infer_cohort_sync(payload: BatchTelemetryRequest):
    """Engine 1: Nightly batch update for dynamic cohorts."""
    if not engine_1_cohort:
        raise HTTPException(status_code=503, detail="Cohort Engine offline.")

    # 1. Load data from Node.js
    df_batch = pd.DataFrame([t.dict() for t in payload.data])

    # 2. Raw Sklearn Math (BGM Clustering)
    X_scaled = engine_1_cohort.scaler.fit_transform(df_batch.values)
    engine_1_cohort.model.fit(X_scaled)
    labels = engine_1_cohort.model.predict(X_scaled)

    # 3. Extract the centroids to find the High Performer cluster
    centroids = engine_1_cohort.scaler.inverse_transform(engine_1_cohort.model.means_)
    active_clusters = np.unique(labels)

    # Assuming Column 0 is Weekly XP, we find the cluster with max XP
    hp_cluster = max(active_clusters, key=lambda c: centroids[c, 0])

    # 4. Format the output targets for Node.js
    targets = [{
        "target_xp": int(centroids[hp_cluster, 0]),
        "target_attendance": round(centroids[hp_cluster, 1], 4),
        "target_on_time": round(centroids[hp_cluster, 2], 4),
        "acceptable_burnout": round(centroids[hp_cluster, 3], 2),
    }]

    return {"status": "success", "new_targets": targets}

@app.post("/api/v1/infer/recommend")
def infer_recommendations(payload: RecommenderRequest):
    if not engine_4_rec or not embedder:
        raise HTTPException(status_code=503, detail="Recommender/Embedder offline.")

    if len(payload.tasks) == 0:
        return {"ranked_task_ids": []}

    hour_sin = math.sin(2 * math.pi * payload.current_hour / 24.0)
    hour_cos = math.cos(2 * math.pi * payload.current_hour / 24.0)

    df_tasks = pd.DataFrame([t.dict() for t in payload.tasks])

    X_infer = pd.DataFrame({
        'time_to_due_date': df_tasks['due_h'],
        'dynamic_difficulty': df_tasks['difficulty'],
        'user_burnout_index': [payload.user_burnout_index] * len(df_tasks),
        'historical_velocity': df_tasks['historical_velocity'],
        'hour_sin': [hour_sin] * len(df_tasks),
        'hour_cos': [hour_cos] * len(df_tasks),
        'temporal_velocity': df_tasks['temporal_velocity']
    })

    base_ltr_scores = engine_4_rec.predict(X_infer)

    internal_trend = get_current_trend()

    task_titles = df_tasks['title'].tolist()
    task_vecs = embedder.encode(task_titles)
    trend_vec = embedder.encode([internal_trend])

    faiss.normalize_L2(task_vecs)
    faiss.normalize_L2(trend_vec)

    semantic_boosts = np.dot(task_vecs, trend_vec.T).flatten()
    final_scores = base_ltr_scores + (np.maximum(semantic_boosts, 0) * 1.5)

    df_tasks['final_score'] = final_scores
    df_sorted = df_tasks.sort_values(by='final_score', ascending=False)

    return {"ranked_task_ids": df_sorted['task_id'].tolist()}

# ==========================================
# 4. RETRAINING ENDPOINTS
# ==========================================

@app.post("/api/v1/retrain/sentiment")
def retrain_sentiment(payload: SentimentRetrainPayload):
    if not payload.training_data:
        raise HTTPException(status_code=400, detail="No training data provided.")

    try:
        df = pd.DataFrame([row.dict() for row in payload.training_data])
        X = df['text']
        y = df['mood_label']

        engine_2_nlp.fit(X, y)

        os.makedirs("output", exist_ok=True)
        with open("output/sentiment_model.pkl", "wb") as f:
            pickle.dump(engine_2_nlp, f)

        return {"status": "success", "message": f"Engine 2 retrained on {len(df)} records!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import requests


# --- Add to Schemas Section ---
class MistralRequest(BaseModel):
    task_list: List[str]


# --- Add to Endpoints Section ---
@app.post("/api/v1/llm/summarize_trend")
def summarize_trend(payload: MistralRequest):
    """Hits the Mistral API to convert raw tasks into a clean syllabus title."""
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY missing from environment.")

    prompt = f"""You are a strict Academic Telemetry Filter. Your job is to extract and format the SINGLE best academic task from a given list.

            INPUT LIST:
            {payload.task_list}

            STRICT PROTOCOL:
            1. FILTER GARBAGE: Completely ignore personal chores ("laundry"), vague goals ("study", "homework"), private names, or non-academic tasks.
            2. ABORT TRIGGER: If every task in the list is filtered out, output EXACTLY one character: ?
            3. SELECT ONE: Do NOT mix or combine tasks together. Pick the single most specific, highest-value academic task remaining.
            4. FORMAT: Return it as a precise syllabus item (e.g., "Physics Sheet 2", "DSP 2025 Midterm").
            5. NO HALLUCINATIONS: Do not invent words.

            Output ONLY the exact formatted task name or ?. No quotes, no markdown."""

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "open-mistral-7b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 15,
        "temperature": 0.0
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="Mistral Rate Limited.")
        elif response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Mistral Error: {response.text}")

        result = response.json()
        title = result['choices'][0]['message']['content'].strip()

        # Failsafe truncate
        if len(title.split()) > 7:
            title = " ".join(title.split()[:5]) + "..."

        return {"trend_title": title}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Add to Schemas Section ---
class ForecastTrainingRow(TimeSeriesRequest):
    target_burnout_3d: float  # The GROUND TRUTH they actually experienced 3 days later


class ForecastRetrainPayload(BaseModel):
    training_data: List[ForecastTrainingRow]


class RecommenderTrainingRow(TaskItem):
    session_id: int  # CRITICAL: XGBRanker needs to group tasks by session
    user_burnout_index: float
    current_hour: int
    user_engaged: int  # 1 if they clicked it, 0 if they ignored it


class RecommenderRetrainPayload(BaseModel):
    training_data: List[RecommenderTrainingRow]


# --- Add to Endpoints Section ---
@app.post("/api/v1/retrain/forecast")
def retrain_forecast(payload: ForecastRetrainPayload):
    """Engine 3 Retraining: Fits the XGBoost Time-Series on real historical lags."""
    if not payload.training_data:
        raise HTTPException(status_code=400, detail="No data provided.")

    try:
        df = pd.DataFrame([row.dict() for row in payload.training_data])

        # XGBoost requires no missing values in the training matrix
        df = df.dropna()
        if df.empty:
            raise HTTPException(status_code=400, detail="Not enough complete 3-day lag sequences to train.")

        y = df['target_burnout_3d']
        X = df.drop(columns=['target_burnout_3d', 'current_burnout'])

        engine_3_ts.fit(X, y)

        with open("output/time_series_model.pkl", "wb") as f:
            pickle.dump(engine_3_ts, f)

        return {"status": "success", "message": f"Engine 3 retrained on {len(df)} historical sequences!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrain/recommend")
def retrain_recommend(payload: RecommenderRetrainPayload):
    """Engine 4 Retraining: Fits XGBRanker using Pairwise Loss and Session Grouping."""
    if not payload.training_data:
        raise HTTPException(status_code=400, detail="No data provided.")

    try:
        df = pd.DataFrame([row.dict() for row in payload.training_data])

        # We must sort by session_id (qid) so XGBRanker knows which tasks were competing against each other
        df = df.sort_values(by='session_id').reset_index(drop=True)

        # Calculate trig features for the training data
        df['hour_sin'] = df['current_hour'].apply(lambda h: math.sin(2 * math.pi * h / 24.0))
        df['hour_cos'] = df['current_hour'].apply(lambda h: math.cos(2 * math.pi * h / 24.0))

        y = df['user_engaged']
        qids = df['session_id']

        X = pd.DataFrame({
            'time_to_due_date': df['due_h'],
            'dynamic_difficulty': df['difficulty'],
            'user_burnout_index': df['user_burnout_index'],
            'historical_velocity': df['historical_velocity'],
            'hour_sin': df['hour_sin'],
            'hour_cos': df['hour_cos'],
            'temporal_velocity': df['temporal_velocity']
        })

        engine_4_rec.fit(X, y, qid=qids, verbose=False)

        with open("output/recommender_engine.pkl", "wb") as f:
            pickle.dump(engine_4_rec, f)

        return {"status": "success",
                "message": f"Engine 4 retrained on {len(df)} tasks across {qids.nunique()} triage sessions!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))