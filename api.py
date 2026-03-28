import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


# ==========================================
# 1. Data Contracts (JSON Schemas)
# ==========================================
class PendingTask(BaseModel):
    task_id: str
    time_to_due_date: float
    dynamic_difficulty: int
    historical_velocity_for_tag: float


class TriageRequest(BaseModel):
    student_id: str
    user_burnout_index: float
    tasks: List[PendingTask]


class RankedTask(BaseModel):
    task_id: str
    predicted_utility: float
    rank: int


class TriageResponse(BaseModel):
    student_id: str
    recommended_action_sequence: List[RankedTask]


# ==========================================
# 2. Service Instantiation & Memory Load
# ==========================================
app = FastAPI(title="AcademaSync LTR Triage Service")

try:
    with open("output/recommender_engine.pkl", "rb") as f:
        ranker_engine = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("[!] FATAL: LTR Engine binary not found in /output.")


# ==========================================
# 3. The Triage Endpoint
# ==========================================
@app.post("/triage_backlog", response_model=TriageResponse)
async def prioritize_tasks(payload: TriageRequest):
    if not payload.tasks:
        raise HTTPException(status_code=400, detail="Task backlog is empty.")

    try:
        # Construct the localized Query Group (qid)
        # The XGBRanker expects a 2D matrix representing the current session
        feature_matrix = []
        task_ids = []

        for task in payload.tasks:
            task_ids.append(task.task_id)
            feature_matrix.append({
                'time_to_due_date': task.time_to_due_date,
                'dynamic_difficulty': task.dynamic_difficulty,
                'user_burnout_index': payload.user_burnout_index,  # Constant across the session
                'historical_velocity_for_tag': task.historical_velocity_for_tag
            })

        df_session = pd.DataFrame(feature_matrix)

        # Enforce exact feature alignment with the serialized champion
        expected_features = [
            'time_to_due_date',
            'dynamic_difficulty',
            'user_burnout_index',
            'historical_velocity_for_tag'
        ]
        X_infer = df_session[expected_features]

        # Execute LambdaRank inference
        utility_scores = ranker_engine.predict(X_infer)

        # Compile and sort the topological sequence
        results = pd.DataFrame({
            'task_id': task_ids,
            'utility': utility_scores
        })

        results = results.sort_values(by='utility', ascending=False).reset_index(drop=True)
        results['rank'] = results.index + 1

        # Construct strictly typed JSON response
        ranked_sequence = [
            RankedTask(
                task_id=row['task_id'],
                predicted_utility=round(float(row['utility']), 4),
                rank=int(row['rank'])
            ) for _, row in results.iterrows()
        ]

        return TriageResponse(
            student_id=payload.student_id,
            recommended_action_sequence=ranked_sequence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LTR Inference Failure: {str(e)}")