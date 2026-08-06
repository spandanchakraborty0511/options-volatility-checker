"""
Phase 6 Validation Script: GARCH(1,1) Forecasting Layer
Fits GARCH(1,1) model to underlying SPY returns, forecasts 21d realized vol, and compares with options ATM IV.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import garch_model

def run_phase6_test(target_date: str = "2023-01-09"):
    print(f"--- Phase 6 Test: GARCH(1,1) Volatility Forecast for {target_date} ---")
    
    garch_summary = garch_model.compare_garch_vs_option_iv(target_date)
    
    print("\nGARCH Model Results:")
    print(f"  Target Date: {garch_summary['date']}")
    print(f"  Options 30d ATM IV: {garch_summary['atm_iv']*100:.2f}%")
    print(f"  GARCH(1,1) 21d Forecast Vol: {garch_summary['garch_forecast_vol']*100:.2f}%")
    print(f"  Macro Volatility Premium (IV - GARCH): {garch_summary['macro_vol_premium']*100:+.2f}%")
    print(f"  Macro Signal: {garch_summary['macro_signal']}")
    
    details = garch_summary['garch_details']
    print(f"  GARCH Persistence (alpha + beta): {details['persistence']:.4f}")
    print(f"  GARCH Parameters: omega={details['garch_params']['omega']:.6e}, alpha={details['garch_params']['alpha']:.4f}, beta={details['garch_params']['beta']:.4f}")
    
    assert garch_summary['garch_forecast_vol'] > 0.0, "GARCH forecasted vol must be positive!"
    print("\nSUCCESS: All Phase 6 GARCH Forecasting assertions passed!")
    return garch_summary

if __name__ == "__main__":
    run_phase6_test()
