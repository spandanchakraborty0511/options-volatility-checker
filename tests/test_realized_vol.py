"""
Phase 2 Validation Script: Realized Volatility & IV Rank Module
Validates close-to-close RV calculations, IV Rank/Percentile, and generates time-series plot.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import realized_vol

def run_phase2_test(target_date: str = "2023-01-09"):
    print(f"--- Phase 2 Test: Volatility Metrics for {target_date} ---")
    
    # 1. Underlying price history
    df_und = realized_vol.extract_underlying_history(end_date=target_date)
    print(f"Extracted {len(df_und)} days of SPY underlying price history.")
    assert len(df_und) >= 252, "Expected at least 252 trading days for IV Rank test!"
    
    # 2. RV calculation
    df_rv = realized_vol.calculate_close_to_close_rv(df_und, window=21)
    current_rv = df_rv['rv_close_to_close'].iloc[-1]
    print(f"21-Day Realized Volatility on {target_date}: {current_rv * 100:.2f}%")
    assert not pd.isna(current_rv), "RV should not be NaN!"
    
    # 3. ATM IV series and IV Rank
    start_lookback = str(int(target_date[:4]) - 1) + target_date[4:]
    df_atm = realized_vol.extract_daily_atm_iv(start_date=start_lookback, end_date=target_date)
    print(f"Extracted {len(df_atm)} days of historical ATM IV data.")
    
    df_rank = realized_vol.calculate_iv_rank_and_percentile(df_atm, window=252)
    latest_rank = df_rank.iloc[-1]
    print(f"ATM IV on {target_date}: {latest_rank['atm_iv'] * 100:.2f}%")
    print(f"52-Week IV Range: [{latest_rank['iv_min_52w']*100:.2f}%, {latest_rank['iv_max_52w']*100:.2f}%]")
    print(f"52-Week IV Rank: {latest_rank['iv_rank']:.2f}%")
    print(f"52-Week IV Percentile: {latest_rank['iv_percentile']:.2f}%")
    
    # 4. Volatility Summary dictionary test
    summary = realized_vol.get_volatility_summary_for_date(target_date)
    print("\nVolatility Summary Dict:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
            
    assert 0.0 <= summary['iv_rank_52w'] <= 100.0, "IV Rank should be bounded in [0, 100]!"
    print("\nSUCCESS: All Phase 2 Realized Volatility & IV Rank assertions passed!")
    
    # 5. Plot RV vs IV Time-Series
    output_plot_path = os.path.join(config.PROJECT_DIR, "tests", "phase2_rv_iv_plot.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    # Merge RV and ATM IV
    df_merged = pd.merge(df_rv, df_rank, on='date', how='inner')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    # Subplot 1: IV vs RV
    ax1.plot(df_merged['date'], df_merged['atm_iv'] * 100, label='ATM Implied Volatility (30d)', color='darkblue', linewidth=1.8)
    ax1.plot(df_merged['date'], df_merged['rv_close_to_close'] * 100, label='Realized Volatility (21d)', color='crimson', linewidth=1.5, linestyle='--')
    ax1.set_ylabel("Volatility (%)")
    ax1.set_title(f"SPY Volatility History — ATM Implied Vol vs 21d Realized Vol ({target_date})")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: IV Rank & IV Percentile
    ax2.plot(df_merged['date'], df_merged['iv_rank'], label='52-Week IV Rank (%)', color='purple', linewidth=1.5)
    ax2.plot(df_merged['date'], df_merged['iv_percentile'], label='52-Week IV Percentile (%)', color='teal', linewidth=1.5, linestyle=':')
    ax2.set_ylabel("Rank / Percentile (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(-5, 105)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print(f"Visual time-series plot saved to: {output_plot_path}")

if __name__ == "__main__":
    run_phase2_test()
