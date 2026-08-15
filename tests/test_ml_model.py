"""
Integration & Validation Unit Test for ML Surface Forecaster & Predictive Alpha Engine.
"""

import sys
import os
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import data_loader
import ml_inference
from api.index import app

class TestMLSurfaceModel(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.test_date = "2023-12-29"
        self.test_expiry = "2024-01-04"

    def test_01_ml_inference_predict_next_day_svi(self):
        res = ml_inference.predict_next_day_svi(self.test_date)
        self.assertIn('predicted_params', res)
        p = res['predicted_params']
        self.assertIn('a', p)
        self.assertIn('b', p)
        self.assertIn('rho', p)
        self.assertIn('m', p)
        self.assertIn('sigma', p)
        self.assertGreater(p['b'], 0)
        self.assertGreater(p['sigma'], 0)
        print("✅ ML Next-Day SVI Parameter Prediction Verified:", p)

    def test_02_ml_alpha_signal_generation(self):
        raw_slice = data_loader.load_raw_options_slice(self.test_date, self.test_expiry)
        clean_slice = data_loader.clean_options_slice(raw_slice)
        self.assertFalse(clean_slice.empty)

        res = ml_inference.predict_next_day_svi(self.test_date)
        df_flagged = ml_inference.generate_ml_alpha_signals(clean_slice, res['predicted_params'], vol_threshold=0.015)
        
        self.assertIn('ml_forecast_iv', df_flagged.columns)
        self.assertIn('ml_signal', df_flagged.columns)
        signals = df_flagged['ml_signal'].unique()
        print("✅ ML Alpha Signals Generated:", signals)

    def test_03_ml_forecast_api_endpoint(self):
        response = self.app.get(f'/api/ml_forecast?date={self.test_date}&expiry={self.test_expiry}')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('curves', data)
        self.assertIn('today_iv', data['curves'])
        self.assertIn('pred_iv', data['curves'])
        self.assertIn('market_points', data)
        print("✅ /api/ml_forecast REST API Endpoint Verified (Status 200 OK)")

if __name__ == '__main__':
    unittest.main()
