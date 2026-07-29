import sys
import os
import io
import pytest
from pathlib import Path

def run_suite():
    project_root = Path(__file__).parent.parent.resolve()
    os.chdir(project_root)
    log_file = project_root / "test_results.log"
    
    # Redirect stdout to both stdout and log_file
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    
    print("="*80)
    print("EXECUTING SPECTRAMAP AUTOMATED TEST SUITE")
    print("="*80)
    
    code = pytest.main([
        "-v",
        "tests",
        "--tb=short"
    ])
    
    print("\nPytest Exit Code:", code)
    return code

if __name__ == "__main__":
    run_suite()
