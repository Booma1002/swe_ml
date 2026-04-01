import requests
import json

BASE_URL = "http://localhost:8001/api/v1"


def run_test(name, endpoint, payload):
    print(f"\n[{name}] -> POST {endpoint}")
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=payload)
        if response.status_code == 200:
            print("✅ SUCCESS!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ FAILED (Status {response.status_code})")
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("🚨 CONNECTION ERROR: Is Uvicorn running on port 8001?")


# ==========================================
# 1. INFERENCE PAYLOADS
# ==========================================
sentiment_payload = {
    "journal_text": "I am so overwhelmed with the Data Structures project. I keep getting segmentation faults and I haven't slept."
}

forecast_payload = {
    "current_burnout": 6.5,
    "sleep_hours_lag_1": 5.0, "sleep_hours_lag_2": 4.5, "sleep_hours_lag_3": 6.0,
    "pomodoro_hours_lag_1": 4.0, "pomodoro_hours_lag_2": 5.5, "pomodoro_hours_lag_3": 3.0,
    "tasks_in_queue_lag_1": 12, "tasks_in_queue_lag_2": 10, "tasks_in_queue_lag_3": 8,
    "days_to_exam_lag_1": 3, "days_to_exam_lag_2": 4, "days_to_exam_lag_3": 5,
    "burnout_index_lag_1": 6.0, "burnout_index_lag_2": 5.5, "burnout_index_lag_3": 4.0
}

cohort_payload = {
    "data": [
        {"weekly_xp_velocity": 450.0, "attendance_rate": 0.95, "on_time_completion_rate": 0.88, "average_burnout_index": 3.5},
        {"weekly_xp_velocity": 420.0, "attendance_rate": 0.92, "on_time_completion_rate": 0.85, "average_burnout_index": 3.8},
        {"weekly_xp_velocity": 120.0, "attendance_rate": 0.60, "on_time_completion_rate": 0.40, "average_burnout_index": 8.2},
        {"weekly_xp_velocity": 110.0, "attendance_rate": 0.55, "on_time_completion_rate": 0.35, "average_burnout_index": 8.5},
        {"weekly_xp_velocity": 500.0, "attendance_rate": 1.0, "on_time_completion_rate": 0.92, "average_burnout_index": 4.0},
        {"weekly_xp_velocity": 480.0, "attendance_rate": 0.98, "on_time_completion_rate": 0.90, "average_burnout_index": 3.2}
    ]
}

recommend_payload = {
    "current_hour": 14,
    "user_burnout_index": 4.5,
    "tasks": [
        {"task_id": "uuid-1", "title": "Read Chapter 4 of OS textbook", "due_h": 72.0, "difficulty": 2,
         "historical_velocity": 45.0, "temporal_velocity": 45.0},
        {"task_id": "uuid-2", "title": "Finish Physics Lab Report", "due_h": 12.0, "difficulty": 4,
         "historical_velocity": 90.0, "temporal_velocity": 90.0},
        {"task_id": "uuid-3", "title": "C++ Pointers Assignment", "due_h": 48.0, "difficulty": 5,
         "historical_velocity": 120.0, "temporal_velocity": 120.0}
    ]
}

llm_payload = {
    "task_list": ["do physics sheet 2", "physics lab prep", "study for physics midterm", "do laundry"]
}

# ==========================================
# 2. RETRAINING PAYLOADS
# ==========================================
retrain_sentiment_payload = {
    "training_data": [
        {"text": "I feel completely unstoppable today, crushed all my tasks.", "mood_label": 1},
        {"text": "I am so tired I can barely keep my eyes open in lecture.", "mood_label": 0},
        {"text": "Best day ever, I aced my physics midterm!", "mood_label": 1},
        {"text": "I am burned out and my brain hurts from math.", "mood_label": 0},
        {"text": "Solid study session, made good progress.", "mood_label": 1}
    ] * 10
}

retrain_forecast_payload = {
    "training_data": [
        {
            "target_burnout_3d": 8.5, "current_burnout": 7.0,
            "sleep_hours_lag_1": 4.0, "sleep_hours_lag_2": 4.5, "sleep_hours_lag_3": 5.0,
            "pomodoro_hours_lag_1": 6.0, "pomodoro_hours_lag_2": 5.5, "pomodoro_hours_lag_3": 5.0,
            "tasks_in_queue_lag_1": 15, "tasks_in_queue_lag_2": 12, "tasks_in_queue_lag_3": 10,
            "days_to_exam_lag_1": 2, "days_to_exam_lag_2": 3, "days_to_exam_lag_3": 4,
            "burnout_index_lag_1": 6.5, "burnout_index_lag_2": 6.0, "burnout_index_lag_3": 5.5
        }
    ]
}

retrain_recommend_payload = {
    "training_data": [
        {"session_id": 1, "user_engaged": 1, "user_burnout_index": 5.0, "current_hour": 10, "task_id": "uuid-10",
         "title": "Math Homework", "due_h": 12.0, "difficulty": 3, "historical_velocity": 60.0,
         "temporal_velocity": 60.0},
        {"session_id": 1, "user_engaged": 0, "user_burnout_index": 5.0, "current_hour": 10, "task_id": "uuid-11",
         "title": "Read History Chapter", "due_h": 72.0, "difficulty": 2, "historical_velocity": 45.0,
         "temporal_velocity": 45.0}
    ] * 10
}

# ==========================================
# 3. RUN THE SUITE
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting AcademaSync API Test Suite...")

    # Test Inference
    run_test("ENGINE 2 (Sentiment)", "/infer/sentiment", sentiment_payload)
    run_test("ENGINE 3 (Forecast)", "/infer/forecast", forecast_payload)
    run_test("ENGINE 1 (Cohort Sync)", "/infer/cohort_sync", cohort_payload)
    run_test("ENGINE 4 (Recommender)", "/infer/recommend", recommend_payload)

    # Test Retraining
    run_test("RETRAIN: Sentiment", "/retrain/sentiment", retrain_sentiment_payload)
    run_test("RETRAIN: Forecast", "/retrain/forecast", retrain_forecast_payload)
    run_test("RETRAIN: Recommender", "/retrain/recommend", retrain_recommend_payload)

    # Test LLM (This will fail with 500 if you don't have MISTRAL_API_KEY in your env)
    run_test("LLM (Trend Summarizer)", "/llm/summarize_trend", llm_payload)

    print("\n🏁 Test Suite Complete!")