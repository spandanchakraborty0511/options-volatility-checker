"""
Phase 1 Validation Script: Data Extraction & Cleaning Pipeline
Validates raw vs cleaned SPY options slice and generates visual inspection plot.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import data_loader

def run_phase1_test(test_date: str = "2023-01-09", test_expiry: str = None):
    print(f"--- Phase 1 Test: Date={test_date} ---")
    
    # 1. Available dates check
    dates = data_loader.get_available_dates()
    print(f"Total available trading dates in DB: {len(dates)}")
    assert test_date in dates, f"Test date {test_date} not in database!"
    
    # 2. Expiries check
    expiries = data_loader.get_expiries_for_date(test_date)
    print(f"Expiries for {test_date}: {len(expiries)} expiries ({expiries[0]} to {expiries[-1]})")
    
    if test_expiry is None:
        # Select an expiry with ~30 days to expiry if possible, else 2nd expiry
        test_expiry = expiries[min(3, len(expiries) - 1)]
    
    print(f"Selected Test Expiry: {test_expiry}")
    
    # 3. Load Raw Slice
    raw_df = data_loader.load_raw_options_slice(test_date, test_expiry)
    print(f"Raw slice row count: {len(raw_df)}")
    
    # 4. Clean Slice
    clean_df = data_loader.clean_options_slice(raw_df)
    print(f"Cleaned slice row count: {len(clean_df)}")
    
    # 5. Assertions for correctness
    assert not clean_df.empty, "Cleaned DataFrame should not be empty!"
    assert (clean_df['iv'] <= config.MAX_IV).all(), "Found IV > MAX_IV in cleaned data!"
    assert (clean_df['bid'] >= config.MIN_BID).all(), "Found bid < MIN_BID in cleaned data!"
    assert (clean_df['ask'] > clean_df['bid']).all(), "Found ask <= bid in cleaned data!"
    assert (clean_df['volume'] >= config.MIN_VOLUME).all(), "Found volume < MIN_VOLUME in cleaned data!"
    assert (clean_df['raw_moneyness'] >= config.MONEYNESS_MIN).all(), "Found strike outside moneyness band!"
    assert (clean_df['raw_moneyness'] <= config.MONEYNESS_MAX).all(), "Found strike outside moneyness band!"
    
    print("SUCCESS: All Phase 1 Data Cleaning assertions passed!")
    
    # 6. Save visual comparison plot
    output_plot_path = os.path.join(config.PROJECT_DIR, "tests", "phase1_cleaning_plot.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot raw vs cleaned calls & puts
    raw_calls = raw_df[raw_df['option_type'] == 'C']
    raw_puts = raw_df[raw_df['option_type'] == 'P']
    clean_calls = clean_df[clean_df['option_type'] == 'C']
    clean_puts = clean_df[clean_df['option_type'] == 'P']
    
    plt.scatter(raw_calls['strike'], raw_calls['iv'] * 100, color='red', alpha=0.3, label='Raw Calls (Filtered Out)', s=20)
    plt.scatter(raw_puts['strike'], raw_puts['iv'] * 100, color='darkred', alpha=0.3, label='Raw Puts (Filtered Out)', marker='x', s=20)
    
    plt.scatter(clean_calls['strike'], clean_calls['iv'] * 100, color='blue', alpha=0.9, label='Clean Calls', s=35)
    plt.scatter(clean_puts['strike'], clean_puts['iv'] * 100, color='green', alpha=0.9, label='Clean Puts', marker='x', s=35)
    
    underlying = raw_df['underlying'].iloc[0]
    plt.axvline(underlying, color='black', linestyle='--', label=f'SPY Spot (${underlying:.2f})')
    
    plt.title(f"SPY Options IV Smile — Raw vs Cleaned ({test_date}, Expiry: {test_expiry}, DTE: {clean_df['dte'].iloc[0]}d)")
    plt.xlabel("Strike Price ($)")
    plt.ylabel("Implied Volatility (%)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print(f"Visual comparison plot saved to: {output_plot_path}")
    return clean_df

if __name__ == "__main__":
    run_phase1_test()
