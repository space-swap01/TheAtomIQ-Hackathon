"""
Root-level Master Pipeline Runner.
Delegates to QuantumBank_Term_Deposit_Optimizer/run_pipeline.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "QuantumBank_Term_Deposit_Optimizer"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import run_pipeline

if __name__ == "__main__":
    run_pipeline.run_full_pipeline()
