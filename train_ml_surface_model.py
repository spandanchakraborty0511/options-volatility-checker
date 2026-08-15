"""
Model Training Script for ML-SVI Volatility Surface Forecaster.
Trains multi-output Random Forest regressor on 14-year feature dataset and exports ml_surface_model.joblib.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def train_model():
    print("Starting ML Surface Model Training...", flush=True)
    start_time = time.time()
    
    dataset_path = os.path.join(config.PROJECT_DIR, "ml_surface_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Feature dataset {dataset_path} not found! Run extract_ml_features.py first.")
        
    df = pd.read_csv(dataset_path)
    print(f"Loaded feature dataset with {len(df)} samples.", flush=True)
    
    feature_cols = [
        'svi_a_lag1', 'svi_b_lag1', 'svi_rho_lag1', 'svi_m_lag1', 'svi_sigma_lag1',
        'svi_a_lag2', 'svi_b_lag2', 'svi_rho_lag2', 'svi_m_lag2', 'svi_sigma_lag2',
        'svi_a_lag5', 'svi_b_lag5', 'svi_rho_lag5', 'svi_m_lag5', 'svi_sigma_lag5',
        'rv_close_to_close_5d', 'rv_close_to_close_21d', 'rv_63d',
        'ret_1d', 'ret_5d', 'ret_21d'
    ]
    
    target_cols = [
        'target_svi_a', 'target_svi_b', 'target_svi_rho', 'target_svi_m', 'target_svi_sigma'
    ]
    
    X = df[feature_cols].values
    Y = df[target_cols].values
    
    # Chronological Out-of-Sample Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}", flush=True)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, Y_train)
    
    # Out-of-Sample Evaluation
    Y_pred = model.predict(X_test)
    
    r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(5)]
    rmse_scores = [np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i])) for i in range(5)]
    mae_scores = [mean_absolute_error(Y_test[:, i], Y_pred[:, i]) for i in range(5)]
    
    param_names = ['a (Level)', 'b (Slope)', 'rho (Skew)', 'm (Shift)', 'sigma (Smooth)']
    
    print("\n--- Out-of-Sample Test Set Model Metrics ---")
    for i in range(5):
        print(f"  {param_names[i]}: R² = {r2_scores[i]:.4f} | RMSE = {rmse_scores[i]:.5f} | MAE = {mae_scores[i]:.5f}")
        
    avg_r2 = float(np.mean(r2_scores))
    print(f"\nAverage Out-of-Sample R² Score: {avg_r2:.4f}")
    
    # Export Model Binary
    model_export_path = os.path.join(config.PROJECT_DIR, "ml_surface_model.joblib")
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'avg_r2': avg_r2,
        'r2_scores': r2_scores
    }, model_export_path)
    
    file_size_mb = os.path.getsize(model_export_path) / (1024 * 1024)
    print(f"Trained model exported to {model_export_path} ({file_size_mb:.2f} MB) in {time.time() - start_time:.2f}s", flush=True)

if __name__ == "__main__":
    train_model()
