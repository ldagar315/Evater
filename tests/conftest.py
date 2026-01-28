import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

# Prefer backend package layout (`backend/app/...`) after upstream restructure.
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
