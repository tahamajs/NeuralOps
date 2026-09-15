import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with (ROOT / "variables.json").open("r", encoding="utf-8") as f:
    vars_data = json.load(f)

print("Loaded variables for A3 module:", sorted(vars_data.keys())[:10])
