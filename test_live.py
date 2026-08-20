from pathlib import Path
import os, json
PROJECT_DIR = Path(r"C:\Users\Hadi Schumacher\SVD_Project")
WORKDIR = PROJECT_DIR / "Codes/framework"
RESULTS_DIR = WORKDIR / "experiments"
print(f"RESULTS_DIR = {RESULTS_DIR}")
if RESULTS_DIR.exists():
    dirs = sorted(RESULTS_DIR.iterdir(), key=os.path.getmtime, reverse=True)
    for d in dirs:
        if d.is_dir():
            hist_file = d / "history" / "log_history.json"
            print(f"Checking {hist_file} - exists? {hist_file.exists()}")
            if hist_file.exists():
                hist = json.loads(hist_file.read_text(encoding="utf-8"))
                loss_history = []
                for h in hist:
                    if "loss" in h:
                        loss_history.append(h)
                print(f"Found {len(loss_history)} loss items")
            break
