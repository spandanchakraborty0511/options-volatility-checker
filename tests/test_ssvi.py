"""
Phase 4 Validation Script: SSVI Multi-Expiry Surface Calibration
Fits SSVI surface across all expiries for a test date, verifies calendar arbitrage freedom, and plots surface curves.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import data_loader
from ssvi_model import SSVISurface

def run_phase4_test(test_date: str = "2023-01-09"):
    print(f"--- Phase 4 Test: SSVI Multi-Expiry Surface for {test_date} ---")
    
    # Load all options for date
    raw_df = data_loader.load_raw_options_slice(test_date)
    clean_df = data_loader.clean_options_slice(raw_df)
    
    num_expiries = clean_df['expiry'].nunique()
    print(f"Loaded {len(clean_df)} clean option points across {num_expiries} expiries for {test_date}.")
    assert num_expiries >= 5, f"Expected at least 5 expiries for SSVI test, got {num_expiries}"
    
    # Fit SSVI Surface
    ssvi = SSVISurface()
    ssvi.fit(clean_df)
    
    print("\nFitted SSVI Global Parameters:")
    print(f"  rho (Global Skew): {ssvi.rho:.4f}")
    print(f"  eta (Curvature Scale): {ssvi.eta:.4f}")
    print(f"  gamma (Term Power): {ssvi.gamma:.4f}")
    print(f"  Surface RMSE: {ssvi.rmse * 100:.2f}% IV")
    print(f"  Surface MAE: {ssvi.mae * 100:.2f}% IV")
    
    # Verify No-Calendar-Arbitrage
    is_arb_free, viols = ssvi.check_calendar_arbitrage()
    print(f"\nCalendar Arbitrage Check: {'PASSED (Arbitrage Free)' if is_arb_free else f'FAILED ({viols} violations)'}")
    assert is_arb_free, f"SSVI surface contained {viols} calendar arbitrage violations!"
    
    print("SUCCESS: All Phase 4 SSVI Surface Calibration assertions passed!")
    
    # Save visual multi-curve term structure smile plot
    output_plot_path = os.path.join(config.PROJECT_DIR, "tests", "phase4_ssvi_surface_plot.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    plt.figure(figsize=(11, 7))
    
    # Select 6 representative expiries across term structure
    df_exp = ssvi.expiries_df.sort_values('tau')
    selected_indices = np.linspace(0, len(df_exp) - 1, 6, dtype=int)
    sample_expiries = df_exp.iloc[selected_indices]
    
    spot = clean_df['spot'].iloc[0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sample_expiries)))
    
    for idx, (_, exp_row) in enumerate(sample_expiries.iterrows()):
        exp_name = exp_row['expiry']
        tau = exp_row['tau']
        dte = exp_row['dte']
        
        slice_df = clean_df[clean_df['expiry'] == exp_name]
        k_smooth = np.linspace(slice_df['log_moneyness'].min(), slice_df['log_moneyness'].max(), 100)
        strikes_smooth = spot * np.exp(k_smooth)
        
        iv_fit = ssvi.predict_iv(k_smooth, tau=tau, theta=exp_row['theta'])
        
        plt.scatter(slice_df['strike'], slice_df['iv'] * 100, color=colors[idx], alpha=0.6, s=25)
        plt.plot(strikes_smooth, iv_fit * 100, label=f'{exp_name} ({dte}d DTE)', color=colors[idx], linewidth=1.8)
        
    plt.axvline(spot, color='gray', linestyle='--', label=f'SPY Spot (${spot:.2f})')
    plt.title(f"SSVI Multi-Expiry Volatility Surface ({test_date}, {num_expiries} Expiries)")
    plt.xlabel("Strike Price ($)")
    plt.ylabel("Implied Volatility (%)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print(f"Visual SSVI surface plot saved to: {output_plot_path}")

if __name__ == "__main__":
    run_phase4_test()
