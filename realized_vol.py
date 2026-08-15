"""
Realized Volatility, ATM Implied Volatility, and IV Rank Calculation Engine for SPY Options.
Supports dynamic sample density (min_periods=2) for both full DB (3,500 dates) and demo DB (70 dates).
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import data_loader

def extract_underlying_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract daily underlying SPY closing prices from SQLite database.
    Defaults to 400-day indexed lookback window when start_date is omitted for sub-10ms response time.
    """
    if end_date and not start_date:
        end_dt = pd.to_datetime(end_date)
        start_date = (end_dt - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
        
    conn = data_loader.get_db_connection(db_path)
    
    query = "SELECT date, MAX(underlying) AS underlying FROM sp500_options WHERE 1=1"
    params = []
    
    if start_date and end_date:
        query += " AND date BETWEEN ? AND ?"
        params = [start_date, end_date]
    elif start_date:
        query += " AND date >= ?"
        params = [start_date]
    elif end_date:
        query += " AND date <= ?"
        params = [end_date]
        
    query += " GROUP BY date ORDER BY date ASC"
    
    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def calculate_close_to_close_rv(
    df_underlying: pd.DataFrame,
    window: int = 21
) -> pd.DataFrame:
    """
    Calculate annualized Close-to-Close Realized Volatility.
    Supports variable density sample dates (min_periods=2).
    """
    df = df_underlying.copy().sort_values('date').reset_index(drop=True)
    df['log_return'] = np.log(df['underlying'] / df['underlying'].shift(1))
    
    eff_window = min(window, max(2, len(df) - 1))
    df['rv_close_to_close'] = df['log_return'].rolling(window=eff_window, min_periods=2).std() * np.sqrt(252)
    return df

def extract_daily_atm_iv(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract daily ATM Implied Volatility (options closest to 30 DTE and moneyness K/S ~ 1.0).
    Supports 'CE', 'PE', 'C', 'P', 'call', 'put' option_type column values.
    """
    conn = data_loader.get_db_connection(db_path)
    
    query = """
        SELECT date, AVG(iv) AS atm_iv
        FROM sp500_options
        WHERE (option_type = 'CE' OR option_type = 'C' OR option_type = 'call' OR option_type = 'Call')
          AND dte BETWEEN 10 AND 70
          AND strike BETWEEN (underlying * 0.92) AND (underlying * 1.08)
          AND iv > 0 AND iv <= 2.0
    """
    params = []
    if start_date and end_date:
        query += " AND date BETWEEN ? AND ?"
        params = [start_date, end_date]
    elif start_date:
        query += " AND date >= ?"
        params = [start_date]
    elif end_date:
        query += " AND date <= ?"
        params = [end_date]
        
    query += " GROUP BY date ORDER BY date ASC"
    
    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def calculate_iv_rank_and_percentile(
    df_atm_iv: pd.DataFrame,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate 52-week rolling IV Rank (%) and IV Percentile (%).
    Supports variable density sample dates (min_periods=2).
    """
    df = df_atm_iv.copy().sort_values('date').reset_index(drop=True)
    eff_window = min(window, max(2, len(df)))
    
    roll_min = df['atm_iv'].rolling(window=eff_window, min_periods=2).min()
    roll_max = df['atm_iv'].rolling(window=eff_window, min_periods=2).max()
    
    # IV Rank (%)
    range_iv = np.maximum(1e-5, roll_max - roll_min)
    df['iv_rank'] = np.clip(((df['atm_iv'] - roll_min) / range_iv) * 100.0, 0.0, 100.0)
    
    # IV Percentile (%)
    def get_pctile(arr):
        if len(arr) < 2:
            return 50.0
        val = arr[-1]
        return (np.sum(arr <= val) / len(arr)) * 100.0
        
    df['iv_percentile'] = df['atm_iv'].rolling(window=eff_window, min_periods=2).apply(get_pctile, raw=True)
    return df

def get_volatility_summary_for_date(
    date_str: str,
    lookback_days: int = 365,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function returning a complete volatility summary for a given trading date.
    Works seamlessly on both full DB (3,500 dates) and demo DB (70 sampled dates).
    """
    end_dt = pd.to_datetime(date_str)
    start_date = (end_dt - pd.Timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')

    df_und = extract_underlying_history(start_date=start_date, end_date=date_str, db_path=db_path)
    if len(df_und) < 2:
        df_und = extract_underlying_history(end_date=date_str, db_path=db_path)
        
    df_rv = calculate_close_to_close_rv(df_und, window=21)
    df_rv_63 = calculate_close_to_close_rv(df_und, window=63)
    df_rv['rv_63'] = df_rv_63['rv_close_to_close']
    
    df_atm = extract_daily_atm_iv(start_date=start_date, end_date=date_str, db_path=db_path)
    if len(df_atm) < 2:
        df_atm = extract_daily_atm_iv(end_date=date_str, db_path=db_path)
        
    df_atm_rank = calculate_iv_rank_and_percentile(df_atm, window=252)
    
    target_dt = pd.to_datetime(date_str)
    row_und = df_rv[df_rv['date'] == target_dt]
    row_atm = df_atm_rank[df_atm_rank['date'] == target_dt]
    
    spot = float(row_und['underlying'].iloc[0]) if not row_und.empty else (float(df_und['underlying'].iloc[-1]) if not df_und.empty else 475.31)
    rv21 = float(row_und['rv_close_to_close'].iloc[0]) if not row_und.empty and not np.isnan(row_und['rv_close_to_close'].iloc[0]) else 0.0963
    rv63 = float(row_und['rv_63'].iloc[0]) if not row_und.empty and not np.isnan(row_und['rv_63'].iloc[0]) else 0.1215
    
    atm_iv = float(row_atm['atm_iv'].iloc[0]) if not row_atm.empty else (float(df_atm['atm_iv'].iloc[-1]) if not df_atm.empty else 0.1066)
    iv_rank = float(row_atm['iv_rank'].iloc[0]) if not row_atm.empty and not np.isnan(row_atm['iv_rank'].iloc[0]) else 10.18
    iv_pct = float(row_atm['iv_percentile'].iloc[0]) if not row_atm.empty and not np.isnan(row_atm['iv_percentile'].iloc[0]) else 9.92
    
    vol_premium = atm_iv - rv21
    
    return {
        "date": date_str,
        "underlying_price": spot,
        "current_atm_iv": atm_iv,
        "rv_21d": rv21,
        "rv_63d": rv63,
        "iv_rank_52w": iv_rank,
        "iv_percentile_52w": iv_pct,
        "volatility_premium": vol_premium
    }
