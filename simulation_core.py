import time
import random
import pandas as pd
import numpy as np
import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer
from collections import Counter
from dotenv import load_dotenv
import requests # Bulletproof REST API
import streamlit as st # Brings the logs to the UI!

load_dotenv()
class EntropyEngine:
    def __init__(self):
        self.subjects = ["the OS memory allocator", "C++ pointers", "the physics lab", "data structures",
                         "backpropagation"]
        self.low_b = (["Felt sharp today.", "Energized."], ["crushed", "optimized"])
        self.med_b = (["Standard grind.", "A bit foggy."], ["wrestled with", "made slow progress on"])
        self.high_b = (["Completely exhausted.", "Running on empty."], ["stared blankly at", "failed compiling"])
        self.anomalies = [
            (0.85, "None", 1.0, 0.0),
            (0.05, "Caught a terrible flu. Couldn't get out of bed.", 0.0, +1.5),
            (0.05, "Hit a massive flow state and coded for 6 hours straight!", 2.5, -1.0),
            (0.05, "Laptop crashed and corrupted my project file. Devastated.", -0.5, +2.0)
        ]

    def roll(self, burnout):
        roll = random.random()
        cumulative = 0.0
        for prob, text, xp_mod, burn_mod in self.anomalies:
            cumulative += prob
            if roll < cumulative:
                if text == "None":
                    subj = np.random.choice(self.subjects)
                    if burnout < 4.0:
                        return f"{np.random.choice(self.low_b[0])} {np.random.choice(self.low_b[1])} {subj}.", 1.0, 0.0
                    elif burnout < 7.5:
                        return f"{np.random.choice(self.med_b[0])} {np.random.choice(self.med_b[1])} {subj}.", 1.0, 0.0
                    else:
                        return f"{np.random.choice(self.high_b[0])} {np.random.choice(self.high_b[1])} {subj}.", 1.0, 0.0
                return text, xp_mod, burn_mod
        return "Standard day.", 1.0, 0.0


class StudentAgent:
    def __init__(self, agent_id, base_xp, base_burnout):
        self.id = agent_id
        self.xp = base_xp
        self.burnout = base_burnout
        self.forecasted_burnout = base_burnout
        self.status = "Normal 🟢"
        self.active_task = "Initializing..."
        self.todays_journal = ""

        # EXTREME TELEMETRY: Tracking all 15 features required by Engine 3
        self.history = {
            'day': [], 'burnout': [], 'xp': [], 'nlp_delta': [],
            'sleep_hours': [], 'pomodoro_hours': [],
            'tasks_in_queue': [], 'days_to_exam': []
        }

        self.total_classes = np.random.randint(40, 60)
        self.classes_attended = int(self.total_classes * np.random.uniform(0.7, 0.95))
        self.classes_late = np.random.randint(0, 5)
        self.classes_absent = self.total_classes - self.classes_attended - self.classes_late

        self.focus_sessions = np.random.randint(50, 150)
        # Macro vs Micro Attendance
        self.classes_cancelled = int(self.classes_absent * 0.2)  # Macro factor
        self.classes_skipped = self.classes_absent - self.classes_cancelled  # Micro factor

        self.unbroken_pomodoros = np.random.randint(10, 30)
        self.avg_session_min = np.random.randint(20, 45)
        self.daily_focus_h = 0.0
        self.sleep_hours_today = 7.0
        self.days_to_exam = np.random.randint(5, 30)  # Dynamic countdown

        self.tasks_done_today = 0
        self.tasks_total_today = np.random.randint(3, 6)
        self.current_streak = np.random.randint(0, 15)
        self.longest_streak = self.current_streak + np.random.randint(0, 10)

        self.curriculum = ['OS Memory Allocator', 'Physics Lab Report', 'Data Structures (Trees)',
                           'Machine Learning (XGBoost)', 'Linear Algebra', 'C++ Pointers']

        self.backlog = pd.DataFrame([
            {"Task_ID": np.random.choice(self.curriculum),
             "Due_H": np.random.randint(12, 120),
             "Diff": np.random.randint(1, 6),
             "Base_LTR": 0.0,
             "Final_Score": 0.0} for _ in range(15)
        ])

    def record_history(self, day, delta, sleep, pomodoro, queue, exam):
        self.history['day'].append(day)
        self.history['burnout'].append(self.burnout)
        self.history['xp'].append(self.xp)
        self.history['nlp_delta'].append(delta)
        self.history['sleep_hours'].append(sleep)
        self.history['pomodoro_hours'].append(pomodoro)
        self.history['tasks_in_queue'].append(queue)
        self.history['days_to_exam'].append(exam)

class MasterSimulation:
    def __init__(self, num_agents=200):
        self.day = 1
        self.agents = [StudentAgent(i, np.random.randint(50, 250),
                                    np.random.uniform(2.0, 7.0)) for i in range(num_agents)]

        self.entropy = EntropyEngine()
        self.global_trend = "..??.."
        self.llm_log = "System Booted. Waiting for time advance..."

        print("[*] STRICT MODE: Loading Elite ML Engines...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # STRICT PICKLE LOADING
        if not os.path.exists("output/continuous_cohort_model.pkl"): raise RuntimeError(
            "FATAL: Engine 1 (Cohort) Pickle missing.")
        with open("output/continuous_cohort_model.pkl", "rb") as f:
            self.engine_1_cohort = pickle.load(f)

        if not os.path.exists("output/sentiment_model.pkl"): raise RuntimeError(
            "FATAL: Engine 2 (Sentiment) Pickle missing.")
        with open("output/sentiment_model.pkl", "rb") as f:
            self.engine_2_nlp = pickle.load(f)

        if not os.path.exists("output/time_series_model.pkl"): raise RuntimeError(
            "FATAL: Engine 3 (Time-Series) Pickle missing.")
        with open("output/time_series_model.pkl", "rb") as f:
            self.engine_3_ts = pickle.load(f)

        if not os.path.exists("output/recommender_engine.pkl"): raise RuntimeError(
            "FATAL: Engine 4 (Recommender) Pickle missing.")
        with open("output/recommender_engine.pkl", "rb") as f:
            self.engine_4_rec = pickle.load(f)

    def synthesize_trend_title(self, task_chunk):
        """Calls Mistral via pure REST API and sends Toasts to the UI."""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            st.toast("❌ MISTRAL_API_KEY missing from .env!", icon="🚨")
            return "API_ERROR"

        st.toast(f"Firing API Request to Mistral for chunk: {task_chunk[0]}...", icon="🌐")

        # --- THE ENHANCED, HALLUCINATION-PROOF PROMPT ---
        prompt = f"""You are a strict Academic Telemetry Filter. Your job is to extract and format the SINGLE best academic task from a given list.
                
                INPUT LIST:
                {task_chunk}
                
                STRICT PROTOCOL:
                1. FILTER GARBAGE: Completely ignore personal chores ("laundry"), vague goals ("study", "homework"), private names, or non-academic tasks.
                2. ABORT TRIGGER: If every task in the list is filtered out, output EXACTLY one character: ?
                3. SELECT ONE: Do NOT mix or combine tasks together. Pick the single most specific, highest-value academic task remaining.
                4. FORMAT (QUANTIFIED SPECIFICITY): Return it as a precise syllabus item (e.g., "Physics Sheet 2", "DSP 2025 Midterm", "Lecture 4: OS Memory"). If it has a number, keep it.
                5. NO HALLUCINATIONS: Do not invent words, subjects, or acronyms not present in the original input.
                
                EXAMPLES:
                Input: ['buy groceries', 'sleep', 'Data Structures (Trees)', 'clean room']
                Output: Data Structures (Trees)
                
                Input: ['study hard', 'do math', 'homework']
                Output: ?
                
                Input: ['Machine Learning (XGBoost)', 'Linear Algebra', 'C++ Pointers']
                Output: Machine Learning (XGBoost)
                
                Output ONLY the exact formatted task name or ?. No quotes, no markdown, no explanations."""

        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "open-mistral-7b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 15,
            "temperature": 0.0  # <--- SET TO 0.0 TO KILL CREATIVITY AND HALLUCINATIONS
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=5)

            if response.status_code == 429:
                return "API_ERROR: 429 Rate Limited (Mistral blocked us)"
            elif response.status_code != 200:
                return f"API_ERROR: HTTP {response.status_code} - {response.text[:50]}"

            result = response.json()
            title = result['choices'][0]['message']['content'].strip()

            if len(title.split()) > 7:
                title = " ".join(title.split()[:5]) + "..."

            return title

        except requests.exceptions.Timeout:
            return "API_ERROR: 5-Second Timeout Exceeded"
        except Exception as e:
            return f"API_ERROR: Network Fatal - {str(e)}"

    def tick(self, skip_llm=False):
        journals = []
        is_weekend = (self.day % 7) in [0, 6]  # Day 6 and 0 are weekends (Macro Variable)

        for a in self.agents:
            text, xp_mod, burn_mod = self.entropy.roll(a.burnout)
            a.todays_journal = text
            if is_weekend and a.burnout > 4.0:
                burn_mod -= 0.5
            a.burnout = np.clip(a.burnout + burn_mod, 1.0, 10.0)
            a.tasks_total_today = np.random.randint(3, 7)

            if xp_mod == 0.0:
                a.status = "Sick 🤒"
                a.classes_absent += 1
                a.classes_skipped += 1
                a.tasks_done_today = 0
            else:
                if xp_mod > 1.5:
                    a.status = "Flow State ⚡"
                    a.daily_focus_h = np.random.uniform(4.0, 6.0)
                    a.tasks_done_today = min(a.tasks_total_today, np.random.randint(3, 6))
                elif a.burnout > 7.5:
                    a.status = "Burned Out 📉"
                    a.classes_late += 1 if random.random() > 0.5 else 0
                    a.daily_focus_h = np.random.uniform(0.5, 2.0)
                    a.tasks_done_today = np.random.randint(0, 2)
                    a.current_streak = 0
                else:
                    a.status = "Normal 🟢"
                    a.classes_attended += 1
                    a.daily_focus_h = np.random.uniform(2.0, 4.0)
                    a.tasks_done_today = np.random.randint(2, 5)
                    a.current_streak += 1
                    if a.current_streak > a.longest_streak: a.longest_streak = a.current_streak

            a.total_classes = a.classes_attended + a.classes_late + a.classes_absent
            a.focus_sessions += int(a.daily_focus_h * 2)
            journals.append(text)

        # STRICT ENGINE 2 (Sentiment)
        probs = self.engine_2_nlp.predict_proba(journals)
        deltas = [(p[0] - p[1]) * 1.5 for p in probs]

        # ==========================================
        # REAL INTEGRATION: ENGINE 3 (TIME-SERIES)
        # ==========================================
        for i, a in enumerate(self.agents):
            a.burnout = np.clip(a.burnout + deltas[i], 1.0, 10.0)

            # Simulate Exam Countdown and Sleep Chaos
            a.days_to_exam -= 1
            if a.days_to_exam <= 0: a.days_to_exam = 30

            if a.status == "Sick 🤒" or a.status == "Burned Out 📉":
                a.sleep_hours_today = np.random.uniform(4.0, 6.0)  # Poor sleep
            else:
                a.sleep_hours_today = np.random.uniform(6.5, 9.0)  # Healthy sleep

            # Record the 15-feature state
            a.record_history(self.day, deltas[i], a.sleep_hours_today, a.daily_focus_h, len(a.backlog), a.days_to_exam)

            if self.engine_3_ts and len(a.history['burnout']) >= 3:
                # PERFECT 15-FEATURE ALIGNMENT FOR XGBOOST
                X_temporal = pd.DataFrame([{
                    'sleep_hours_lag_1': a.history['sleep_hours'][-1],
                    'sleep_hours_lag_2': a.history['sleep_hours'][-2],
                    'sleep_hours_lag_3': a.history['sleep_hours'][-3],
                    'pomodoro_hours_lag_1': a.history['pomodoro_hours'][-1],
                    'pomodoro_hours_lag_2': a.history['pomodoro_hours'][-2],
                    'pomodoro_hours_lag_3': a.history['pomodoro_hours'][-3],
                    'tasks_in_queue_lag_1': a.history['tasks_in_queue'][-1],
                    'tasks_in_queue_lag_2': a.history['tasks_in_queue'][-2],
                    'tasks_in_queue_lag_3': a.history['tasks_in_queue'][-3],
                    'days_to_exam_lag_1': a.history['days_to_exam'][-1],
                    'days_to_exam_lag_2': a.history['days_to_exam'][-2],
                    'days_to_exam_lag_3': a.history['days_to_exam'][-3],
                    'burnout_index_lag_1': a.history['burnout'][-1],
                    'burnout_index_lag_2': a.history['burnout'][-2],
                    'burnout_index_lag_3': a.history['burnout'][-3]
                }])
                # Strict 15-Feature Inference
                a.forecasted_burnout = float(np.clip(self.engine_3_ts.predict(X_temporal)[0], 1.0, 10.0))
            elif len(a.history['burnout']) >= 3:
                a.forecasted_burnout = np.clip((a.history['burnout'][-1] * 0.6) + (a.history['burnout'][-2] * 0.3) + (
                            a.history['burnout'][-3] * 0.1) + (deltas[i] * 0.5), 1.0, 10.0)
            else:
                a.forecasted_burnout = a.burnout

        trend_vec = self.embedder.encode([self.global_trend])
        faiss.normalize_L2(trend_vec)

        # STRICT ENGINE 4 (Recommender)
        for a in self.agents:
            if a.status == "Sick 🤒":
                a.active_task = "Bedridden"
                continue

            a.backlog['Due_H'] -= 24

            X_infer = pd.DataFrame({
                'time_to_due_date': a.backlog['Due_H'],
                'dynamic_difficulty': a.backlog['Diff'],
                'user_burnout_index': [a.burnout] * len(a.backlog),
                'historical_velocity': [60.0] * len(a.backlog),
                'hour_sin': [0.5] * len(a.backlog),
                'hour_cos': [0.86] * len(a.backlog),
                'temporal_velocity': [60.0] * len(a.backlog)
            })
            a.backlog['Base_LTR'] = self.engine_4_rec.predict(X_infer)

            task_vecs = self.embedder.encode(a.backlog['Task_ID'].tolist())
            faiss.normalize_L2(task_vecs)
            sims = np.dot(task_vecs, trend_vec.T).flatten()

            # --- RESTORED: Calculate Final Score and Assign Task ---
            a.backlog['Final_Score'] = a.backlog['Base_LTR'] + (np.maximum(sims, 0) * 1.5)
            a.backlog = a.backlog.sort_values('Final_Score', ascending=False)
            a.active_task = a.backlog.iloc[0]['Task_ID']

            # --- NEW XP GAMIFICATION SYSTEM ---
            # 1. Task Completion XP (+15 per task)
            task_xp = a.tasks_done_today * 15

            # 2. Pomodoro Session XP (+5 per chunk)
            pomodoros_today = int(a.daily_focus_h * 2)  # Assuming 30m chunks
            pomodoro_xp = pomodoros_today * 5

            # 3. Unbroken Flow Bonus (+20 for every session not exited early)
            unbroken_today = int(pomodoros_today * np.random.uniform(0.4, 0.9)) if pomodoros_today > 0 else 0
            a.unbroken_pomodoros += unbroken_today
            bonus_xp = unbroken_today * 20

            # Apply Total Daily XP
            daily_total_xp = int((task_xp + pomodoro_xp + bonus_xp) * (0.2 if a.burnout > 7.5 else 1.0))
            a.xp += daily_total_xp
            # --- DEQUEUE COMPLETED TASKS ---
            if a.tasks_done_today > 0 and not a.backlog.empty:
                # Slice off the top N tasks that the student actually finished today
                tasks_to_remove = min(a.tasks_done_today, len(a.backlog))
                a.backlog = a.backlog.iloc[tasks_to_remove:].reset_index(drop=True)

            # --- REPLENISH BACKLOG (The Infinite Semester) ---
            # If their queue gets too small, the professors assign new homework
            if len(a.backlog) < 10:
                new_tasks = pd.DataFrame([
                    {
                        "Task_ID": np.random.choice(a.curriculum),
                        "Due_H": np.random.randint(48, 168),
                        "Diff": np.random.randint(1, 6),
                        "Base_LTR": 0.0,
                        "Final_Score": 0.0} for _ in range(5)
                ])
                a.backlog = pd.concat([a.backlog, new_tasks], ignore_index=True)

        # STRICT ENGINE 1 (Cohort BGM)
        df_batch = pd.DataFrame({
            'weekly_xp_velocity': [a.xp for a in self.agents],
            'attendance_rate': [a.classes_attended / max(1, a.total_classes) for a in self.agents],
            'on_time_completion_rate': [0.8] * len(self.agents),
            'average_burnout_index': [a.burnout for a in self.agents]
        })

        # We enforce the use of the StandardScaler inside your custom Engine 1 class
        X_scaled = self.engine_1_cohort.scaler.fit_transform(df_batch.values)
        self.engine_1_cohort.model.fit(X_scaled)
        labels = self.engine_1_cohort.model.predict(X_scaled)

        centroids = self.engine_1_cohort.scaler.inverse_transform(self.engine_1_cohort.model.means_)

        # FIX: Only look for the max XP among clusters that ACTUALLY have students in them
        active_clusters = np.unique(labels)
        hp_cluster = max(active_clusters, key=lambda c: centroids[c, 0])

        hp_tasks = [a.active_task for i, a in enumerate(self.agents) if
                    labels[i] == hp_cluster and a.active_task != "Bedridden"]

        # FAILSAFE: If the entire high-performing cluster is sick, just take the top 10% of students
        if not hp_tasks:
            hp_tasks = [a.active_task for a in sorted(self.agents, key=lambda x: x.xp, reverse=True)[:20] if
                        a.active_task != "Bedridden"]

        if hp_tasks:
            sorted_tasks = [task for task, count in Counter(hp_tasks).most_common()]
            if skip_llm:
                # ⏩ FAST FORWARD: Assign the raw topmost task to mimic AI interaction instantly
                self.global_trend = sorted_tasks[0]
                self.llm_log = f"[Day {self.day}] ⏩ FAST FORWARD: Trend auto-assigned to raw task '{self.global_trend}' (API Muted)."
            else:
                # 🧠 NO CACHE: Force an LLM call on every single button click
                chunk_size = 5
                trend_result = "?"

                log_msgs = [f"[Day {self.day}] 🔄 User Click Triggered! Engaging Mistral API..."]

                for i in range(0, len(sorted_tasks), chunk_size):
                    task_chunk = sorted_tasks[i: i + chunk_size]
                    log_msgs.append(f"> Scanning Chunk {i // chunk_size + 1}: {task_chunk[0]}...")

                    trend_result = self.synthesize_trend_title(task_chunk)

                    if trend_result.startswith("API_ERROR"):
                        log_msgs.append(f"🛑 {trend_result}")
                        log_msgs.append("Halting LLM loop. Using raw topmost task instead.")
                        trend_result = sorted_tasks[0]  # Fallback to raw task
                        break
                    elif trend_result != "?":
                        log_msgs.append(f"✅ Success! Generated Title: '{trend_result}'")
                        break
                    else:
                        log_msgs.append("⚠️ Chunk rejected (Privacy/Vague). Throttling 3s before next chunk...")
                        time.sleep(3)

                if trend_result == "?":
                    log_msgs.append("⚠️ All chunks failed. Applying raw topmost task.")
                    self.global_trend = sorted_tasks[0]
                else:
                    self.global_trend = trend_result

                self.llm_log = "\n".join(log_msgs)

        self.day += 1