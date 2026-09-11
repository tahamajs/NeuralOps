import os
import sys
import csv
import json
import shutil
from pathlib import Path

REPO_DIR = Path(r"D:\SVD_Project") if os.path.exists(r"D:\SVD_Project") else Path(__file__).resolve().parent
RESULTS_DIR = REPO_DIR / "results"
STATE_FILE = REPO_DIR / "finish_benchmark_state.json"
MODELS = ["distilbert", "roberta", "gpt2"]
PASS_THRESHOLD = 56.0

print(f"[Recheck] Scanning results in {RESULTS_DIR}...", flush=True)

# 1. Discover all modules from runner files
runners = sorted(REPO_DIR.glob("run_all_4_*.py"))
all_modules = [r.stem[len("run_all_4_"):] for r in runners]

state = {}
if STATE_FILE.exists():
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        state = {}

requeued_modules = []
fully_ok_modules = []

for mod in all_modules:
    unified_dir = RESULTS_DIR / f"unified_{mod}"
    mod_ok = True
    reason = []

    if not unified_dir.exists():
        mod_ok = False
        reason.append("no_results_dir")
    else:
        for model in MODELS:
            csv_path = unified_dir / model / "summary.csv"
            if not csv_path.exists():
                mod_ok = False
                reason.append(f"missing_{model}_csv")
                continue
            
            best_acc = 0.0
            best_status = "FAIL"
            try:
                with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        m_name = (row.get("module_name") or "").strip()
                        if m_name == "full":
                            continue
                        try:
                            acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0)) or 0)
                            st = row.get("status", "FAIL")
                            if acc > best_acc:
                                best_acc, best_status = acc, st
                        except Exception:
                            pass
            except Exception as e:
                mod_ok = False
                reason.append(f"{model}_csv_read_error")
                continue

            # Check criteria
            if best_status != "PASS" or best_acc < PASS_THRESHOLD:
                mod_ok = False
                reason.append(f"{model}_{best_status}_{best_acc:.2f}%")
            elif abs(best_acc - 50.92) < 0.05 or abs(best_acc - 50.0) < 0.05:
                # Chance level on RoBERTa or GPT-2
                if model in ("roberta", "gpt2"):
                    mod_ok = False
                    reason.append(f"{model}_chance_level_{best_acc:.2f}%")

    if mod_ok:
        fully_ok_modules.append(mod)
    else:
        requeued_modules.append((mod, ", ".join(reason)))
        # Delete stale summary.csv and finetuned_model.pt for failing pairs so runner cannot skip
        if unified_dir.exists():
            for model in MODELS:
                model_dir = unified_dir / model
                if model_dir.exists():
                    for stale_file in [
                        model_dir / "summary.csv",
                        model_dir / "finetuned_model.pt",
                        model_dir / "summary.csv.bak"
                    ]:
                        if stale_file.exists():
                            try:
                                stale_file.unlink()
                            except Exception:
                                pass
        # Remove from state file so finish_benchmark re-runs it
        if mod in state:
            del state[mod]

# Save cleaned state
STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

print(f"[Recheck] Total modules found: {len(all_modules)}")
print(f"[Recheck] Fully OK on all 3 models: {len(fully_ok_modules)}")
print(f"[Recheck] Re-queued for run: {len(requeued_modules)}")
for m, r in requeued_modules[:30]:
    print(f"  - {m}: {r}")
if len(requeued_modules) > 30:
    print(f"  ... and {len(requeued_modules) - 30} more.")
