import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Ensure tests behave consistently if callers set ENV vars globally.
os.environ.setdefault("ENV", "test")

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Ensure tests behave consistently if callers set ENV vars globally.
os.environ.setdefault("ENV", "test")
