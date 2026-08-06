"""
Data Loader & Cleaning Pipeline Module for SPY Options Volatility Checker (Phase 1).
Provides database connection, slice querying, and robust data cleaning functions.
Supports local full database (sp500_data.db) and Vercel cloud demo database (demo_options.db).
"""

import sqlite3
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

import config

def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Return a SQLite connection.
    Automatically chooses sp500_data.db if available locally, else falls back to demo_options.db.
    """
    if db_path and os.path.exists(db_path):
        target_path = db_path
    elif os.path.exists(config.DB_PATH):
        target_path = config.DB_PATH
    else:
        # Fallback for Vercel deployment
        target_path = os.path.join(config.PROJECT_DIR, "demo_options.db")
        
    return sqlite3.connect(target_path)

def get_available_dates(db_path: Optional[str] = None) -> List[str]:
    """
    Get all distinct trading dates available in the database in ascending order.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("SELECT date FROM sp500_options GROUP BY date ORDER BY date ASC")
        return [row[0] for row in cursor.fetchall()]

def get_expiries_for_date(date_str: str, db_path: Optional[str] = None) -> List[str]:
    """
    Get all distinct option expiration dates for a given trading date.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT expiry FROM sp500_options WHERE date = ? GROUP BY expiry ORDER BY expiry ASC",
            (date_str,)
        )
        return [row[0] for row in cursor.fetchall()]

def load_raw_options_slice(
    date_str: str,
    expiry_str: Optional[str] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load raw options data for a specific date (and optional specific expiry).
    Returns raw DataFrame directly as stored in SQLite.
    """
    with get_db_connection(db_path) as conn:
        if expiry_str:
            query = """
                SELECT date, expiry, strike, option_type, underlying, price, bid, ask, volume, iv, dte
                FROM sp500_options
                WHERE date = ? AND expiry = ?
                ORDER BY strike ASC, option_type ASC
            """
            params = (date_str, expiry_str)
        else:
            query = """
                SELECT date, expiry, strike, option_type, underlying, price, bid, ask, volume, iv, dte
                FROM sp500_options
                WHERE date = ?
                ORDER BY expiry ASC, strike ASC, option_type ASC
            """
            params = (date_str,)
            
        df = pd.read_sql(query, conn, params=params)
    return df

def clean_options_slice(
    df_raw: pd.DataFrame,
    moneyness_min: float = config.MONEYNESS_MIN,
    moneyness_max: float = config.MONEYNESS_MAX,
    min_volume: float = config.MIN_VOLUME,
    min_bid: float = config.MIN_BID,
    require_ask_gt_bid: bool = config.REQUIRE_ASK_GT_BID,
    max_iv: float = config.MAX_IV,
    min_iv: float = config.MIN_IV,
    min_dte: int = config.MIN_DTE,
    max_dte: int = config.MAX_DTE
) -> pd.DataFrame:
    """
    Applies data quality and liquidity filters to raw options DataFrame.
    """
    if df_raw.empty:
        return df_raw.copy()
    
    df = df_raw.copy()
    
    df = df.dropna(subset=['strike', 'underlying', 'iv', 'bid', 'ask', 'dte'])
    df = df[(df['dte'] >= min_dte) & (df['dte'] <= max_dte)]
    
    df['spot'] = df['underlying']
    df['raw_moneyness'] = df['strike'] / df['spot']
    df = df[(df['raw_moneyness'] >= moneyness_min) & (df['raw_moneyness'] <= moneyness_max)]
    
    if min_volume > 0 and 'volume' in df.columns:
        df = df[df['volume'].notna() & (df['volume'] >= min_volume)]
        
    df = df[df['bid'] >= min_bid]
    if require_ask_gt_bid:
        df = df[df['ask'] > df['bid']]
        
    df = df[(df['iv'] >= min_iv) & (df['iv'] <= max_iv)]
    
    if df.empty:
        return df
        
    df['tau'] = df['dte'] / 365.0
    df['forward'] = df['spot']
    df['log_moneyness'] = np.log(df['strike'] / df['forward'])
    df['total_variance'] = (df['iv'] ** 2) * df['tau']
    df['mid_price'] = (df['bid'] + df['ask']) / 2.0
    df['bid_ask_spread'] = df['ask'] - df['bid']
    df['relative_spread'] = df['bid_ask_spread'] / df['mid_price']
    
    return df
