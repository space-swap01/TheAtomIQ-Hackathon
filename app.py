"""
Root-level Streamlit App.
Delegates to QuantumBank_Term_Deposit_Optimizer/app.py
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "QuantumBank_Term_Deposit_Optimizer"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Execute app.py from QuantumBank_Term_Deposit_Optimizer
app_file = APP_DIR / "app.py"
with open(app_file, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(app_file), 'exec')
    exec(code, {'__name__': '__main__', '__file__': str(app_file)})
