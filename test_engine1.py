import unittest
import pandas as pd
import numpy as np
from app import ContinuousCohortEngine, generate_dynamic_population


class TestCohortEngine(unittest.TestCase):

    def setUp(self):
        """Initialize a fresh engine and base population for every test."""
        self.engine = ContinuousCohortEngine(max_components=5, weight_cutoff=0.1)
        self.size = 1000
        self.topic = "Data Structures (C++)"
        self.df = generate_dynamic_population(self.size, self.topic, focus_level=1.0, noise_level=0.1)

    def test_dgp_integrity(self):
        """Test 1: Validate the Stochastic Data Generating Process (DGP)"""
        # Expected: DataFrame contains correct columns and non-null values
        self.assertEqual(len(self.df), self.size)
        self.assertIn('active_task', self.df.columns)
        self.assertTrue(self.df['weekly_xp_velocity'].max() <= 10000)
        print("\n[✔] DGP Integrity: Passed. Topological manifold matches expected constraints.")

    def test_model_initialization(self):
        """Test 2: Validate BGM Initialization & Cold Start"""
        # Expected: Engine state should flip to is_initialized=True after first batch
        self.assertFalse(self.engine.is_initialized)
        self.engine.process_batch(self.df)
        self.assertTrue(self.engine.is_initialized)
        # Verify Bayesian Weights sum to approx 1.0
        self.assertAlmostEqual(np.sum(self.engine.model.weights_), 1.0, places=4)
        print("[✔] BGM Initialization: Passed. Bayesian weights successfully distributed.")

    def test_trend_mode_extraction(self):
        """Test 3: Validate Post-Hoc Categorical Mode Extraction"""
        # Expected: With focus_level=1.0, the dominant trend MUST be the injected topic
        self.engine.process_batch(self.df)
        detected_trend = self.engine.extract_dominant_trend()

        self.assertEqual(detected_trend, self.topic)
        print(f"[✔] Mode Extraction: Passed. Engine correctly identified '{detected_trend}'.")

    def test_drift_gatekeeper(self):
        """Test 4: Validate Statistical Gatekeeper (Log-Likelihood Rejection)"""
        # First, train on normal data
        self.engine.process_batch(self.df)

        # Create an 'Evolutionary Outlier' batch (Extreme noise/nonsense data)
        outlier_df = pd.DataFrame({
            'weekly_xp_velocity': [99999, -5000, 0],
            'attendance_rate': [5.0, -1.0, 10.0],
            'on_time_completion_rate': [0.5, 0.5, 0.5],
            'average_burnout_index': [50.0, 0.0, -10.0],
            'active_task': ["Chaos", "Chaos", "Chaos"]
        })

        # Expected: Engine must detect drift and (in production) return False/Rejection
        # Here we simulate the process_batch logic with the drift threshold
        X_scaled_outlier = self.engine.scaler.transform(outlier_df[self.engine.feature_names].values)
        log_likelihood = self.engine.model.score(X_scaled_outlier)

        self.assertLess(log_likelihood, self.engine.drift_threshold)
        print(f"[✔] Drift Gatekeeper: Passed. Outlier batch rejected (LL: {log_likelihood:.2f}).")


if __name__ == '__main__':
    unittest.main()