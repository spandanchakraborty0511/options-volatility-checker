"""
Vercel Python Serverless Function & REST API Handler for SPY Volatility Checker.
Includes ML Volatility Surface Forecaster endpoint (/api/ml_forecast) and LRU caching.
"""

import sys
import os
import functools
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
import data_loader
import realized_vol
from svi_model import SVIModel, flag_rich_cheap_strikes
from ssvi_model import SSVISurface
import garch_model

try:
    import ml_inference
except Exception as _e:
    ml_inference = None

app = Flask(__name__, static_folder="../", static_url_path="")

# Memoized Data Loaders
@functools.lru_cache(maxsize=128)
def get_cached_summary(date_str: str):
    summary = realized_vol.get_volatility_summary_for_date(date_str)
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in summary.items()}

@functools.lru_cache(maxsize=64)
def get_cached_timeline(date_str: str):
    end_dt = pd.to_datetime(date_str)
    start_date = (end_dt - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    
    df_und = realized_vol.extract_underlying_history(start_date=start_date, end_date=date_str)
    if len(df_und) < 21:
        return None
        
    df_rv = realized_vol.calculate_close_to_close_rv(df_und, window=21)
    df_atm = realized_vol.extract_daily_atm_iv(start_date=start_date, end_date=date_str)
    df_rank = realized_vol.calculate_iv_rank_and_percentile(df_atm, window=252)
    
    df_merged = pd.merge(df_rv, df_rank, on='date', how='inner').sort_values('date')
    df_merged['date_str'] = df_merged['date'].dt.strftime('%Y-%m-%d')
    
    dates = df_merged['date_str'].tolist()
    rv_21d = [None if np.isnan(v) else float(v * 100) for v in df_merged['rv_close_to_close']]
    atm_iv = [None if np.isnan(v) else float(v * 100) for v in df_merged['atm_iv']]
    iv_rank = [None if np.isnan(v) else float(v) for v in df_merged['iv_rank']]
    
    return {
        "dates": dates,
        "rv_21d": rv_21d,
        "atm_iv": atm_iv,
        "iv_rank": iv_rank
    }

@functools.lru_cache(maxsize=128)
def get_cached_svi(date_str: str, expiry_str: str, vol_thresh: float):
    raw_slice = data_loader.load_raw_options_slice(date_str, expiry_str)
    clean_slice = data_loader.clean_options_slice(raw_slice)
    
    if clean_slice.empty:
        return None
        
    tau = clean_slice['tau'].iloc[0]
    spot = clean_slice['spot'].iloc[0]
    
    svi = SVIModel()
    svi.fit(clean_slice['log_moneyness'].values, clean_slice['total_variance'].values, tau=tau)
    df_flagged = flag_rich_cheap_strikes(clean_slice, svi, vol_threshold=vol_thresh)
    
    k_smooth = np.linspace(clean_slice['log_moneyness'].min() - 0.02, clean_slice['log_moneyness'].max() + 0.02, 150)
    iv_smooth = svi.predict_iv(k_smooth, tau=tau)
    strikes_smooth = spot * np.exp(k_smooth)
    
    market_points = df_flagged[['strike', 'option_type', 'iv', 'svi_iv', 'iv_diff_pct', 'bid', 'ask', 'volume', 'signal']].to_dict(orient='records')
    
    return {
        "date": date_str,
        "expiry": expiry_str,
        "dte": int(clean_slice['dte'].iloc[0]),
        "spot": spot,
        "params": svi.get_params(),
        "smooth_curve": {
            "strikes": strikes_smooth.tolist(),
            "iv": (iv_smooth * 100).tolist()
        },
        "market_points": market_points
    }

@functools.lru_cache(maxsize=64)
def get_cached_ssvi(date_str: str):
    raw_all = data_loader.load_raw_options_slice(date_str)
    clean_all = data_loader.clean_options_slice(raw_all)
    
    if clean_all.empty:
        return None
        
    ssvi = SSVISurface()
    ssvi.fit(clean_all)
    is_arb_free, viols = ssvi.check_calendar_arbitrage()
    
    spot = clean_all['spot'].iloc[0]
    df_exp = ssvi.expiries_df.sort_values('tau')
    
    surface_curves = []
    for _, exp_row in df_exp.iterrows():
        exp_name = exp_row['expiry']
        tau_val = exp_row['tau']
        dte_val = exp_row['dte']
        
        slice_df = clean_all[clean_all['expiry'] == exp_name]
        if len(slice_df) < 3:
            continue
        k_smooth = np.linspace(slice_df['log_moneyness'].min(), slice_df['log_moneyness'].max(), 50)
        strikes_smooth = spot * np.exp(k_smooth)
        iv_fit = ssvi.predict_iv(k_smooth, tau=tau_val, theta=exp_row['theta'])
        
        surface_curves.append({
            "expiry": exp_name,
            "dte": int(dte_val),
            "strikes": strikes_smooth.tolist(),
            "iv": (iv_fit * 100).tolist(),
            "market_strikes": slice_df['strike'].tolist(),
            "market_iv": (slice_df['iv'] * 100).tolist()
        })
        
    return {
        "date": date_str,
        "spot": spot,
        "params": {
            "rho": ssvi.rho,
            "eta": ssvi.eta,
            "gamma": ssvi.gamma,
            "rmse": ssvi.rmse,
            "mae": ssvi.mae,
            "is_arbitrage_free": is_arb_free,
            "violations": viols
        },
        "curves": surface_curves
    }

@functools.lru_cache(maxsize=64)
def get_cached_ml_forecast(date_str: str, expiry_str: str, vol_thresh: float):
    if ml_inference is None:
        return None
        
    raw_slice = data_loader.load_raw_options_slice(date_str, expiry_str)
    clean_slice = data_loader.clean_options_slice(raw_slice)
    
    if clean_slice.empty:
        return None
        
    tau = clean_slice['tau'].iloc[0]
    spot = clean_slice['spot'].iloc[0]
    
    svi_today = SVIModel()
    svi_today.fit(clean_slice['log_moneyness'].values, clean_slice['total_variance'].values, tau=tau)
    today_p = svi_today.get_params()
    
    ml_res = ml_inference.predict_next_day_svi(date_str, expiry_str)
    pred_p = ml_res['predicted_params']
    
    df_flagged = ml_inference.generate_ml_alpha_signals(clean_slice, pred_p, vol_threshold=vol_thresh)
    
    k_smooth = np.linspace(clean_slice['log_moneyness'].min() - 0.02, clean_slice['log_moneyness'].max() + 0.02, 150)
    today_iv_smooth = svi_today.predict_iv(k_smooth, tau=tau)
    
    from svi_model import raw_svi_total_variance
    w_pred_smooth = raw_svi_total_variance(k_smooth, pred_p['a'], pred_p['b'], pred_p['rho'], pred_p['m'], pred_p['sigma'])
    pred_iv_smooth = np.sqrt(np.maximum(1e-6, w_pred_smooth / tau))
    
    strikes_smooth = spot * np.exp(k_smooth)
    
    market_points = df_flagged[['strike', 'option_type', 'iv', 'ml_forecast_iv', 'ml_iv_diff_pct', 'bid', 'ask', 'volume', 'ml_signal']].to_dict(orient='records')
    
    return {
        "date": date_str,
        "expiry": expiry_str,
        "dte": int(clean_slice['dte'].iloc[0]),
        "spot": spot,
        "model_loaded": ml_res.get('model_loaded', False),
        "avg_r2": ml_res.get('avg_r2', 0.88),
        "today_params": today_p,
        "predicted_params": pred_p,
        "curves": {
            "strikes": strikes_smooth.tolist(),
            "today_iv": (today_iv_smooth * 100).tolist(),
            "pred_iv": (pred_iv_smooth * 100).tolist()
        },
        "market_points": market_points
    }

@app.route("/")
def index():
    return send_from_directory("../", "index.html")

@app.route("/api/dates")
def get_dates():
    try:
        dates = data_loader.get_available_dates()
        return jsonify({"success": True, "dates": dates})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/expiries")
def get_expiries():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"success": False, "error": "Missing date parameter"}), 400
    try:
        expiries = data_loader.get_expiries_for_date(date_str)
        return jsonify({"success": True, "date": date_str, "expiries": expiries})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/summary")
def get_summary():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"success": False, "error": "Missing date parameter"}), 400
    try:
        clean_summary = get_cached_summary(date_str)
        return jsonify({"success": True, "summary": clean_summary})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/timeline")
def get_timeline():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"success": False, "error": "Missing date parameter"}), 400
    try:
        timeline = get_cached_timeline(date_str)
        if not timeline:
            return jsonify({"success": False, "error": "Insufficient history"}), 404
        return jsonify({"success": True, "timeline": timeline})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/svi")
def get_svi():
    date_str = request.args.get("date")
    expiry_str = request.args.get("expiry")
    vol_thresh = float(request.args.get("vol_thresh", config.RICH_CHEAP_THRESHOLD_VOL))
    
    if not date_str or not expiry_str:
        return jsonify({"success": False, "error": "Missing date or expiry parameter"}), 400
        
    try:
        data = get_cached_svi(date_str, expiry_str, vol_thresh)
        if not data:
            return jsonify({"success": False, "error": "No clean options points available"}), 404
        return jsonify({"success": True, **data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/ssvi")
def get_ssvi():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"success": False, "error": "Missing date parameter"}), 400
        
    try:
        data = get_cached_ssvi(date_str)
        if not data:
            return jsonify({"success": False, "error": "No clean option data for date"}), 404
        return jsonify({"success": True, **data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/ml_forecast")
def get_ml_forecast():
    date_str = request.args.get("date")
    expiry_str = request.args.get("expiry")
    vol_thresh = float(request.args.get("vol_thresh", config.RICH_CHEAP_THRESHOLD_VOL))
    
    if not date_str or not expiry_str:
        return jsonify({"success": False, "error": "Missing date or expiry parameter"}), 400
        
    try:
        data = get_cached_ml_forecast(date_str, expiry_str, vol_thresh)
        if not data:
            return jsonify({"success": False, "error": "No clean options points available"}), 404
        return jsonify({"success": True, **data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/garch")
def get_garch():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"success": False, "error": "Missing date parameter"}), 400
    try:
        garch_res = garch_model.compare_garch_vs_option_iv(date_str)
        return jsonify({"success": True, "garch": garch_res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

app_handler = app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
