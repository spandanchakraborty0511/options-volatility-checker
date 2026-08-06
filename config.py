"""
Configuration settings and filter thresholds for SPY Options Volatility Checker.
All cleaning constants are centralized here as named parameters.
"""

import os

# Base paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, "sp500_data.db")

# Phase 1 Data Cleaning Thresholds
# Moneyness filter (Strike / Spot ratio): keep strikes within [65%, 135%] of underlying price
MONEYNESS_MIN = 0.65
MONEYNESS_MAX = 1.35

# Liquidity filters
MIN_VOLUME = 1                # Minimum volume (drop 0 or NULL volume rows)
MIN_BID = 0.01                # Minimum bid price (drop non-positive bid)
REQUIRE_ASK_GT_BID = True    # Require ask > bid

# IV Sanity Cap (decimal form, 2.0 = 200% annualized IV)
MAX_IV = 2.0
MIN_IV = 0.005                # 0.5% floor to avoid degenerate zero IVs

# Expiry Filter (days to expiry)
MIN_DTE = 1                   # Exclude 0 DTE to prevent division by zero in annualized total variance
MAX_DTE = 730                 # Exclude > 2 years if extreme

# Model Calibration Settings (Phase 3 & Phase 4)
SVI_MAX_ITER = 1000
RICH_CHEAP_THRESHOLD_VOL = 0.015  # 1.5% IV deviation to flag Rich/Cheap
