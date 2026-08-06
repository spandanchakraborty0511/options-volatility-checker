"""
Phase 3 Validation Script: Single Expiry SVI Model Calibration
Fits Raw SVI to cleaned options slice, checks parameters & RMSE, and generates smile calibration plot with Rich/Cheap signals.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import data_loader
from svi_model import SVIModel, flag_rich_cheap_strikes

def run_phase3_test(test_date: str = "2023-01-09", target_dte: int = 30):
    print(f"--- Phase 3 Test: SVI Calibration for {test_date} ---")
    
    expiries = data_loader.get_expiries_for_date(test_date)
    
    # Pick expiry closest to target_dte (e.g. 30 days)
    best_expiry = expiries[0]
    min_dte_diff = 999
    
    for exp in expiries:
        raw_tmp = data_loader.load_raw_options_slice(test_date, exp)
        if not raw_tmp.empty:
            dte_val = raw_tmp['dte'].iloc[0]
            if abs(dte_val - target_dte) < min_dte_diff:
                min_dte_diff = abs(dte_val - target_dte)
                best_expiry = exp
                
    test_expiry = best_expiry
    print(f"Selected Test Expiry: {test_expiry}")
    
    raw_df = data_loader.load_raw_options_slice(test_date, test_expiry)
    clean_df = data_loader.clean_options_slice(raw_df)
    
    print(f"Cleaned options count: {len(clean_df)} (DTE: {clean_df['dte'].iloc[0]}d)")
    assert len(clean_df) >= 8, "Need at least 8 option points for robust SVI test!"
    
    # Fit SVI Model
    tau = clean_df['tau'].iloc[0]
    svi = SVIModel()
    svi.fit(clean_df['log_moneyness'].values, clean_df['total_variance'].values, tau=tau)
    
    params = svi.get_params()
    print("\nFitted SVI Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.6f}")
        
    # Assert parameter validity and bounds
    assert svi.b >= 0, "SVI parameter b must be non-negative!"
    assert -1.0 < svi.rho < 1.0, "SVI parameter rho must be in (-1, 1)!"
    assert svi.sigma > 0, "SVI parameter sigma must be positive!"
    assert svi.rmse < 0.025, f"SVI RMSE too high: {svi.rmse:.4f} (expected < 2.5% IV error)"
    
    # Flag Rich / Cheap strikes
    df_flagged = flag_rich_cheap_strikes(clean_df, svi)
    
    signal_counts = df_flagged['signal'].value_counts().to_dict()
    print(f"\nRich/Cheap Signal Distribution: {signal_counts}")
    
    print("\nSUCCESS: All Phase 3 SVI Calibration assertions passed!")
    
    # Save visual plot
    output_plot_path = os.path.join(config.PROJECT_DIR, "tests", "phase3_svi_smile_plot.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Generate smooth SVI curve across log-moneyness range
    k_smooth = np.linspace(clean_df['log_moneyness'].min() - 0.02, clean_df['log_moneyness'].max() + 0.02, 200)
    iv_smooth = svi.predict_iv(k_smooth, tau=tau)
    spot = clean_df['spot'].iloc[0]
    strikes_smooth = spot * np.exp(k_smooth)
    
    plt.plot(strikes_smooth, iv_smooth * 100, label=f'Fitted SVI Curve (RMSE: {svi.rmse*100:.2f}%)', color='black', linewidth=2.0)
    
    fair_mask = df_flagged['signal'] == 'FAIR'
    rich_mask = df_flagged['signal'] == 'RICH'
    cheap_mask = df_flagged['signal'] == 'CHEAP'
    
    plt.scatter(df_flagged.loc[fair_mask, 'strike'], df_flagged.loc[fair_mask, 'iv'] * 100, color='blue', alpha=0.8, label='Fair Market IV', s=40)
    plt.scatter(df_flagged.loc[rich_mask, 'strike'], df_flagged.loc[rich_mask, 'iv'] * 100, color='red', alpha=0.9, label='Rich (Overpriced > +1.5%)', marker='^', s=70)
    plt.scatter(df_flagged.loc[cheap_mask, 'strike'], df_flagged.loc[cheap_mask, 'iv'] * 100, color='green', alpha=0.9, label='Cheap (Underpriced < -1.5%)', marker='v', s=70)
    
    plt.axvline(spot, color='gray', linestyle='--', label=f'SPY Spot (${spot:.2f})')
    plt.title(f"SVI Volatility Smile & Rich/Cheap Flags ({test_date}, Expiry: {test_expiry}, DTE: {clean_df['dte'].iloc[0]}d)")
    plt.xlabel("Strike Price ($)")
    plt.ylabel("Implied Volatility (%)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print(f"Visual SVI smile plot saved to: {output_plot_path}")
    return df_flagged

if __name__ == "__main__":
    run_phase3_test()
