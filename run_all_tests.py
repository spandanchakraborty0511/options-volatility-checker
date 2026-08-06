"""
Master Test Runner for SPY Options Volatility Checker.
Executes all validation tests from Phase 1 through Phase 6 and reports results.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_data_loader import run_phase1_test
from tests.test_realized_vol import run_phase2_test
from tests.test_svi import run_phase3_test
from tests.test_ssvi import run_phase4_test
from tests.test_garch import run_phase6_test

def main():
    print("=" * 70)
    print(" SPY OPTIONS VOLATILITY CHECKER -- FULL TEST SUITE")
    print("=" * 70)
    
    start_time = time.time()
    passed = 0
    total = 5
    
    # Phase 1
    try:
        print("\n[Phase 1] Data Extraction & Cleaning Pipeline...")
        run_phase1_test()
        passed += 1
    except Exception as e:
        print(f"[FAIL] Phase 1 Failed: {e}")
        
    # Phase 2
    try:
        print("\n[Phase 2] Realized Volatility & IV Rank Module...")
        run_phase2_test()
        passed += 1
    except Exception as e:
        print(f"[FAIL] Phase 2 Failed: {e}")
        
    # Phase 3
    try:
        print("\n[Phase 3] Single Expiry Raw SVI Calibration...")
        run_phase3_test()
        passed += 1
    except Exception as e:
        print(f"[FAIL] Phase 3 Failed: {e}")
        
    # Phase 4
    try:
        print("\n[Phase 4] Multi-Expiry Surface SVI (SSVI)...")
        run_phase4_test()
        passed += 1
    except Exception as e:
        print(f"[FAIL] Phase 4 Failed: {e}")
        
    # Phase 6
    try:
        print("\n[Phase 6] GARCH(1,1) Volatility Forecasting Layer...")
        run_phase6_test()
        passed += 1
    except Exception as e:
        print(f"[FAIL] Phase 6 Failed: {e}")
        
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f" TEST SUITE RESULT: {passed}/{total} PHASES PASSED ({elapsed:.2f}s elapsed)")
    print("=" * 70)

if __name__ == "__main__":
    main()
