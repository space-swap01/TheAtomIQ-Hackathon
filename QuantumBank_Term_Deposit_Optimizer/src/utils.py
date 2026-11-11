"""
Utility functions, logging configuration, paths, and reproducibility settings
for the QuantumBank Term Deposit Campaign Optimizer.
"""

import os
import sys
import random
import logging
from pathlib import Path
import numpy as np

# Random seed for reproducibility across all stages
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Set global random seeds for Python, NumPy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_project_root() -> Path:
    """Return the base project directory."""
    # If run from inside QuantumBank_Term_Deposit_Optimizer or parent directory
    current = Path.cwd().resolve()
    if (current / "QuantumBank_Term_Deposit_Optimizer").exists():
        return (current / "QuantumBank_Term_Deposit_Optimizer").resolve()
    if current.name == "QuantumBank_Term_Deposit_Optimizer":
        return current
    # Default fallback
    return current

def setup_logger(name: str = "QuantumBank") -> logging.Logger:
    """Configure and return a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
