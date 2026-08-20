from pathlib import Path
import os, json
RESULTS_DIR = Path(r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\experiments")
if RESULTS_DIR.exists():
    for d in sorted(RESULTS_DIR.iterdir(), key=os.path.getmtime, reverse=True):
        if d.is_dir():
            hist_file = d / "history" / "log_history.json"
            print(f"Checking {hist_file}...")
            if hist_file.exists():
                print("Exists!")
                hist = json.loads(hist_file.read_text(encoding="utf-8"))
                for h in hist:
                    if "loss" in h:
                        print("Found loss:", h["loss"])
            break
