"""
Realized Volatility & IV Rank Module (Phase 2).
Calculates historical realized volatility (Close-to-Close, Garman-Klass) from underlying price history
and computes IV Rank / IV Percentile metrics for SPY options.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

import config
import data_loader

def extract_underlying_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract daily SPY underlying price history from the options database.
    Defaults start_date to 730 days prior to end_date if omitted, enabling ultra-fast index range scan.
    """
    if end_date and not start_date:
        end_dt = pd.to_datetime(end_date)
        start_date = (end_dt - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
        
    with data_loader.get_db_connection(db_path) as conn:
        query = "SELECT date, underlying FROM sp500_options"
        params = []
        if start_date and end_date:
            query += " WHERE date BETWEEN ? AND ?"
            params = [start_date, end_date]
        elif start_date:
            query += " WHERE date >= ?"
            params = [start_date]
        elif end_date:
            query += " WHERE date <= ?"
            params = [end_date]
            
        query += " GROUP BY date ORDER BY date ASC"
        df = pd.read_sql(query, conn, params=params)
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def calculate_close_to_close_rv(
    df_underlying: pd.DataFrame,
    window: int = 21,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculate Close-to-Close Realized Volatility (annualized).
    Formula: std(log_returns) * sqrt(252) over rolling window.
    """
    df = df_underlying.copy()
    df['log_return'] = np.log(df['underlying'] / df['underlying'].shift(1))
    
    rolling_std = df['log_return'].rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
    df['rv_close_to_close'] = rolling_std * np.sqrt(trading_days)
    return df

def extract_daily_atm_iv(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_dte: int = 30,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract nearest-to-ATM IV for each trading date around a target DTE (default ~30 days).
    Uses 500-day default lookback if only end_date is specified for fast indexed retrieval.
    """
    if end_date and not start_date:
        end_dt = pd.to_datetime(end_date)
        start_date = (end_dt - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
        
    with data_loader.get_db_connection(db_path) as conn:
        conditions = ["dte BETWEEN ? AND ?", "volume > 0", "bid > 0", "ask > bid", "iv > 0", "iv <= 2.0"]
        params = [target_dte - 15, target_dte + 15]
        
        if start_date and end_date:
            conditions.insert(0, "date BETWEEN ? AND ?")
            params = [start_date, end_date] + params
        elif start_date:
            conditions.insert(0, "date >= ?")
            params = [start_date] + params
        elif end_date:
            conditions.insert(0, "date <= ?")
            params = [end_date] + params
            
        where_clause = " WHERE " + " AND ".join(conditions)
        query = f"""
            SELECT date, strike, underlying, iv, dte, ABS(strike - underlying) as abs_diff
            FROM sp500_options
            {where_clause}
        """
        df = pd.read_sql(query, conn, params=params)
        
    if df.empty:
        return pd.DataFrame(columns=['date', 'atm_iv'])
        
    idx_min = df.groupby('date')['abs_diff'].idxmin()
    df_atm = df.loc[idx_min, ['date', 'iv', 'dte', 'underlying']].copy()
    df_atm['date'] = pd.to_datetime(df_atm['date'])
    df_atm = df_atm.rename(columns={'iv': 'atm_iv'}).sort_values('date').reset_index(drop=True)
    return df_atm

def calculate_iv_rank_and_percentile(
    df_atm_iv: pd.DataFrame,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate 52-week (rolling 252-day) IV Rank and IV Percentile.
    """
    df = df_atm_iv.copy()
    
    rolling_min = df['atm_iv'].rolling(window=window, min_periods=21).min()
    rolling_max = df['atm_iv'].rolling(window=window, min_periods=21).max()
    
    range_iv = rolling_max - rolling_min
    range_iv = np.where(range_iv == 0, np.nan, range_iv)
    
    df['iv_min_52w'] = rolling_min
    df['iv_max_52w'] = rolling_max
    df['iv_rank'] = ((df['atm_iv'] - rolling_min) / range_iv) * 100.0
    
    def pct_rank(vals):
        if len(vals) < 2:
            return np.nan
        current = vals[-1]
        return (vals < current).mean() * 100.0
        
    df['iv_percentile'] = df['atm_iv'].rolling(window=window, min_periods=21).apply(pct_rank, raw=True)
    return df

def get_volatility_summary_for_date(
    date_str: str,
    lookback_days: int = 365,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function returning a complete volatility summary for a given trading date.
    Uses explicit 365-day indexed window for fast sub-second computation.
    """
    end_dt = pd.to_datetime(date_str)
    start_date = (end_dt - pd.Timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')

    df_und = extract_underlying_history(start_date=start_date, end_date=date_str, db_path=db_path)
    if len(df_und) < 21:
        raise ValueError(f"Insufficient underlying price history prior to {date_str}")
        
    df_rv = calculate_close_to_close_rv(df_und, window=21)
    df_rv_63 = calculate_close_to_close_rv(df_und, window=63)
    df_rv['rv_63'] = df_rv_63['rv_close_to_close']
    
    df_atm = extract_daily_atm_iv(start_date=start_date, end_date=date_str, db_path=db_path)
    df_atm_rank = calculate_iv_rank_and_percentile(df_atm, window=252)
    
    target_dt = pd.to_datetime(date_str)
    
    rv_row = df_rv[df_rv['date'] == target_dt]
    atm_row = df_atm_rank[df_atm_rank['date'] == target_dt]
    
    current_rv_21 = rv_row['rv_close_to_close'].values[0] if not rv_row.empty else np.nan
    current_rv_63 = rv_row['rv_63'].values[0] if not rv_row.empty else np.nan
    
    if not atm_row.empty:
        current_atm_iv = atm_row['atm_iv'].values[0]
        iv_rank = atm_row['iv_rank'].values[0]
        iv_percentile = atm_row['iv_percentile'].values[0]
    else:
        current_atm_iv = np.nan
        iv_rank = np.nan
        iv_percentile = np.nan
        
    vol_premium = (current_atm_iv - current_rv_21) if (not np.isnan(current_atm_iv) and not np.isnan(current_rv_21)) else np.nan
    
    return {
        'date': date_str,
        'underlying_price': df_und['underlying'].iloc[-1],
        'current_atm_iv': current_atm_iv,
        'rv_21d': current_rv_21,
        'rv_63d': current_rv_63,
        'volatility_premium': vol_premium,
        'iv_rank_52w': iv_rank,
        'iv_percentile_52w': iv_percentile
    }
