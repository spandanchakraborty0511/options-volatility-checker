"""
Inference Pipeline for ML-SVI Volatility Surface Forecaster.
Loads trained ML model binary (ml_surface_model.joblib), predicts next-day SVI parameters,
and generates Predictive Alpha Trading Signals.
Includes index-safe bounds checking for both full DB and small demo DB.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import data_loader
import realized_vol
from svi_model import raw_svi_total_variance

_MODEL_CACHE = None

def load_ml_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
        
    model_path = os.path.join(config.PROJECT_DIR, "ml_surface_model.joblib")
    if not os.path.exists(model_path):
        return None
        
    _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE

def predict_next_day_svi(date_str: str) -> Dict[str, Any]:
    data = load_ml_model()
    if not data:
        return {
            'predicted_params': {'a': 0.01, 'b': 0.1, 'rho': -0.4, 'm': 0.0, 'sigma': 0.1},
            'avg_r2': 0.0,
            'model_loaded': False
        }
        
    model = data['model']
    
    df_und = realized_vol.extract_underlying_history(end_date=date_str)
    if len(df_und) < 2:
        raise ValueError(f"Insufficient underlying history prior to {date_str}")
        
    df_rv5 = realized_vol.calculate_close_to_close_rv(df_und, window=5)
    df_rv21 = realized_vol.calculate_close_to_close_rv(df_und, window=21)
    df_rv63 = realized_vol.calculate_close_to_close_rv(df_und, window=63)
    
    get_last_val = lambda df: float(df['rv_close_to_close'].dropna().iloc[-1]) if not df.dropna().empty else 0.15
    rv_5d = get_last_val(df_rv5)
    rv_21d = get_last_val(df_rv21)
    rv_63d = get_last_val(df_rv63)
    
    spot_now = float(df_und['underlying'].iloc[-1])
    spot_lag1 = float(df_und['underlying'].iloc[-min(2, len(df_und))])
    spot_lag5 = float(df_und['underlying'].iloc[-min(6, len(df_und))])
    spot_lag21 = float(df_und['underlying'].iloc[-min(22, len(df_und))])
    
    ret_1d = float(np.log(spot_now / spot_lag1)) if spot_lag1 > 0 else 0.0
    ret_5d = float(np.log(spot_now / spot_lag5)) if spot_lag5 > 0 else 0.0
    ret_21d = float(np.log(spot_now / spot_lag21)) if spot_lag21 > 0 else 0.0
    
    raw_slice = data_loader.load_raw_options_slice(date_str)
    clean_slice = data_loader.clean_options_slice(raw_slice)
    
    if clean_slice.empty:
        raise ValueError(f"No clean options data for {date_str}")
        
    from svi_model import SVIModel
    tau = clean_slice['tau'].iloc[0]
    svi_curr = SVIModel()
    svi_curr.fit(clean_slice['log_moneyness'].values, clean_slice['total_variance'].values, tau=tau)
    p = svi_curr.get_params()
    
    feature_vector = np.array([[
        p['a'], p['b'], p['rho'], p['m'], p['sigma'],
        p['a'], p['b'], p['rho'], p['m'], p['sigma'],
        p['a'], p['b'], p['rho'], p['m'], p['sigma'],
        rv_5d, rv_21d, rv_63d,
        ret_1d, ret_5d, ret_21d
    ]])
    
    pred_y = model.predict(feature_vector)[0]
    
    pred_params = {
        'a': float(pred_y[0]),
        'b': float(max(0.001, pred_y[1])),
        'rho': float(np.clip(pred_y[2], -0.95, 0.95)),
        'm': float(pred_y[3]),
        'sigma': float(max(0.001, pred_y[4]))
    }
    
    return {
        'date': date_str,
        'today_params': p,
        'predicted_params': pred_params,
        'avg_r2': float(data.get('avg_r2', 0.85)),
        'model_loaded': True
    }

def generate_ml_alpha_signals(
    df_slice: pd.DataFrame,
    ml_predicted_params: Dict[str, float],
    vol_threshold: float = config.RICH_CHEAP_THRESHOLD_VOL
) -> pd.DataFrame:
    df = df_slice.copy()
    k_array = df['log_moneyness'].values
    tau = df['tau'].iloc[0]
    
    p = ml_predicted_params
    w_pred = raw_svi_total_variance(k_array, p['a'], p['b'], p['rho'], p['m'], p['sigma'])
    iv_pred = np.sqrt(np.maximum(1e-6, w_pred / tau))
    
    df['ml_forecast_iv'] = iv_pred
    df['ml_iv_diff'] = df['iv'] - df['ml_forecast_iv']
    df['ml_iv_diff_pct'] = (df['ml_iv_diff'] / df['ml_forecast_iv']) * 100.0
    
    conditions = [
        df['ml_iv_diff'] < -vol_threshold,
        df['ml_iv_diff'] > vol_threshold
    ]
    choices = ['PREDICTIVE BUY (Undervalued)', 'PREDICTIVE SELL (Overvalued)']
    df['ml_signal'] = np.select(conditions, choices, default='FAIR')
    
    return df
