import sys
from pathlib import Path

# Wipe old summary entries for the timed out modules so they get re-evaluated with the 900s timeout
summary_path = Path("results/final_paper_run/summary.csv")
