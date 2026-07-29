#!/usr/bin/env python3
"""
Automated E2E Test Suite Runner for SpectraMap
==============================================
Runs all test modules across Tier 1, Tier 2, Tier 3, and Tier 4.
Produces a formatted execution summary report.
"""

import sys
import os
import time
import pytest
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.resolve()
    tests_dir = project_root / "tests"
    
    os.chdir(project_root)
    
    print("=" * 80)
    print(" SPECTRAMAP END-TO-END AUTOMATED TEST SUITE RUNNER")
    print("=" * 80)
    print(f" Working Directory: {project_root}")
    print(f" Test Directory   : {tests_dir}")
    print("=" * 80)
    
    start_time = time.time()
    
    pytest_args = [
        "-v",
        str(tests_dir),
        "--tb=short",
        "-W", "ignore::DeprecationWarning",
        "-W", "ignore::UserWarning"
    ]
    
    ret_code = pytest.main(pytest_args)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f" TEST SUITE EXECUTION COMPLETED IN {elapsed:.2f} SECONDS")
    print(f" Pytest Exit Code: {ret_code} ({'SUCCESS' if ret_code == 0 else 'FAILURE'})")
    print("=" * 80)
    
    return ret_code

if __name__ == "__main__":
    sys.exit(main())
