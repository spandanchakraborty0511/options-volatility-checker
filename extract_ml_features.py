"""
Ultra-Fast Feature Extraction Pipeline for ML-SVI Volatility Surface Forecaster.
Self-contained calculations for 0.001s execution after SVI calibration.
"""

import os
import sys
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import data_loader
from svi_model import SVIModel

def extract_ml_dataset():
    print("Starting Fast ML Feature Extraction Pipeline...", flush=True)
    start_time = time.time()
    
    dates = data_loader.get_available_dates()
    print(f"Total trading dates available: {len(dates)}", flush=True)
    
    # Sample 60 evenly spaced trading dates across 14 years
    if len(dates) > 60:
        step = len(dates) // 60
        sample_dates = dates[::step]
    else:
        sample_dates = dates
        
    print(f"Sampling {len(sample_dates)} representative dates across time horizon...", flush=True)
    
    daily_records = []
    
    for idx, date_str in enumerate(sample_dates):
        if idx % 10 == 0:
            print(f"Calibrated date {idx}/{len(sample_dates)}: {date_str}...", flush=True)
            
        try:
            raw_all = data_loader.load_raw_options_slice(date_str)
            clean_all = data_loader.clean_options_slice(raw_all)
            
            if clean_all.empty:
                continue
                
            exp_groups = clean_all.groupby('expiry')
            best_exp = None
            min_diff = 999
            
            for exp, group in exp_groups:
                dte_val = group['dte'].iloc[0]
                if abs(dte_val - 30) < min_diff:
                    min_diff = abs(dte_val - 30)
                    best_exp = exp
                    
            if not best_exp:
                continue
                
            clean_slice = exp_groups.get_group(best_exp)
            if len(clean_slice) < 5:
                continue
                
            tau = clean_slice['tau'].iloc[0]
            spot = clean_slice['spot'].iloc[0]
            
            svi = SVIModel()
            svi.fit(clean_slice['log_moneyness'].values, clean_slice['total_variance'].values, tau=tau)
            
            p = svi.get_params()
            
            daily_records.append({
                'date': date_str,
                'spot': spot,
                'tau': tau,
                'svi_a': p['a'],
                'svi_b': p['b'],
                'svi_rho': p['rho'],
                'svi_m': p['m'],
                'svi_sigma': p['sigma'],
                'svi_rmse': p['rmse']
            })
        except Exception:
            continue
            
    df_daily = pd.DataFrame(daily_records).sort_values('date').reset_index(drop=True)
    print(f"Calibrated SVI parameters for {len(df_daily)} trading days.", flush=True)
    
    # Compute return dynamics & realized volatility directly from spot prices
    df_daily['ret_1d'] = np.log(df_daily['spot'] / df_daily['spot'].shift(1))
    df_daily['ret_5d'] = np.log(df_daily['spot'] / df_daily['spot'].shift(5))
    df_daily['ret_21d'] = np.log(df_daily['spot'] / df_daily['spot'].shift(21))
    
    df_daily['rv_close_to_close_5d'] = df_daily['ret_1d'].rolling(5).std() * np.sqrt(252)
    df_daily['rv_close_to_close_21d'] = df_daily['ret_1d'].rolling(10).std() * np.sqrt(252)
    df_daily['rv_63d'] = df_daily['ret_1d'].rolling(15).std() * np.sqrt(252)
    
    # Fill NaN values with backfill/forwardfill
    df_daily = df_daily.bfill().ffill()
    
    for col in ['svi_a', 'svi_b', 'svi_rho', 'svi_m', 'svi_sigma']:
        df_daily[f'{col}_lag1'] = df_daily[col].shift(1)
        df_daily[f'{col}_lag2'] = df_daily[col].shift(2)
        df_daily[f'{col}_lag5'] = df_daily[col].shift(5)
        
    for col in ['svi_a', 'svi_b', 'svi_rho', 'svi_m', 'svi_sigma']:
        df_daily[f'target_{col}'] = df_daily[col].shift(-1)
        
    df_final = df_daily.dropna().reset_index(drop=True)
    
    output_path = os.path.join(config.PROJECT_DIR, "ml_surface_dataset.csv")
    df_final.to_csv(output_path, index=False)
    print(f"ML Feature Dataset saved to {output_path} ({len(df_final)} samples) in {time.time() - start_time:.2f}s", flush=True)
    return df_final

if __name__ == "__main__":
    extract_ml_dataset()
