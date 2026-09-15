"""
Rank Sweep — Test every rank-based compression module at multiple ranks.
Outputs structured CSV + JSON for comparative analysis.
"""
from __future__ import annotations
import os
import sys
import csv
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

_THIS = Path(__file__).resolve()
_RUNNERS_DIR = _THIS.parent
_EXPERIMENTAL = _RUNNERS_DIR.parent
_CODES_DIR = _EXPERIMENTAL.parent
_PROJECT_ROOT = _CODES_DIR.parent

if os.path.exists(r"D:\svd_project"):
    _PROJECT_ROOT = Path(r"D:\svd_project")
elif os.path.exists(r"D:\SVD_Project"):
    _PROJECT_ROOT = Path(r"D:\SVD_Project")

sys.path.insert(0, str(_CODES_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

# ---- Configuration ----
RANKS = [8, 16, 32, 64, 128, 256, 384]
MODELS = ["distilbert", "roberta", "gpt2"]
RECOVERY_STEPS = 10
TIMEOUT_SECONDS = 900

MANIFEST_PATH = _PROJECT_ROOT / "results" / "rank_sweep" / "module_manifest.json"
OUT_DIR = _PROJECT_ROOT / "results" / "rank_sweep"
CSV_PATH = OUT_DIR / "sweep_results.csv"
JSON_PATH = OUT_DIR / "sweep_results.json"
LOG_DIR = OUT_DIR / "logs"

FIELDS = [
    "module", "model", "requested_rank", "actual_rank", "status",
    "accuracy_pct", "f1_score", "num_compressed_layers", "num_fallback_layers",
    "total_params_before", "total_params_after", "param_ratio",
    "avg_reconstruction_error", "elapsed_s", "error",
]

def load_manifest():
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found: {MANIFEST_PATH}")
        sys.exit(1)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["modules"]

def find_runner(runner_name: str) -> Path | None:
    candidates = [
        _PROJECT_ROOT / "Codes" / "runners" / runner_name,
        _PROJECT_ROOT / runner_name,
        _RUNNERS_DIR / runner_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def read_summary(module: str, model: str):
    candidates = [
        _PROJECT_ROOT / "results" / f"unified_{module}" / model / "summary.csv",
        _PROJECT_ROOT / "results" / module / model / "summary.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                for r in rows:
                    if r.get("module_name") not in ("full", "baseline", ""):
                        return r
                if rows:
                    return rows[-1]
            except Exception as e:
                print(f"[read_summary] Warning reading {p}: {e}")
    return None

def to_float(v, d=0.0):
    try:
        if v is None or v == "" or str(v).lower() == "nan":
            return d
        return float(v)
    except (ValueError, TypeError):
        return d

def run_one(mod: dict, model: str, rank: int) -> dict:
    name = mod["name"]
    runner = find_runner(mod["runner"])
    if runner is None:
        return {
            "module": name, "model": model, "requested_rank": rank,
            "actual_rank": "", "status": "SKIPPED",
            "accuracy_pct": 0.0, "f1_score": 0.0,
            "num_compressed_layers": 0, "num_fallback_layers": 0,
            "total_params_before": "", "total_params_after": "",
            "param_ratio": 1.0, "avg_reconstruction_error": "",
            "elapsed_s": 0.0, "error": f"runner not found: {mod['runner']}",
        }

    rank_flag = mod.get("rank_flag", "--rank")
    cmd = [
        sys.executable, "-u", str(runner),
        "--models", model,
        "--recovery_steps", str(RECOVERY_STEPS),
        rank_flag, str(rank),
    ]

    log_path = LOG_DIR / f"{name}_{model}_rank{rank}.log"
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS, cwd=str(_PROJECT_ROOT)
        )
        elapsed = time.monotonic() - t0
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)
    except subprocess.TimeoutExpired:
        return {
            "module": name, "model": model, "requested_rank": rank,
            "actual_rank": "", "status": "TIMEOUT",
            "accuracy_pct": 0.0, "f1_score": 0.0,
            "num_compressed_layers": 0, "num_fallback_layers": 0,
            "total_params_before": "", "total_params_after": "",
            "param_ratio": 1.0, "avg_reconstruction_error": "",
            "elapsed_s": TIMEOUT_SECONDS, "error": "timeout",
        }
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "module": name, "model": model, "requested_rank": rank,
            "actual_rank": "", "status": "FAIL",
            "accuracy_pct": 0.0, "f1_score": 0.0,
            "num_compressed_layers": 0, "num_fallback_layers": 0,
            "total_params_before": "", "total_params_after": "",
            "param_ratio": 1.0, "avg_reconstruction_error": "",
            "elapsed_s": round(elapsed, 2), "error": str(e),
        }

    summary = read_summary(name, model)
    if summary is None:
        err_msg = "summary.csv missing"
        if proc.returncode != 0:
            err_msg = f"runner failed (exit {proc.returncode})"
        return {
            "module": name, "model": model, "requested_rank": rank,
            "actual_rank": "", "status": "NO_SUMMARY" if proc.returncode == 0 else "FAIL",
            "accuracy_pct": 0.0, "f1_score": 0.0,
            "num_compressed_layers": 0, "num_fallback_layers": 0,
            "total_params_before": "", "total_params_after": "",
            "param_ratio": 1.0, "avg_reconstruction_error": "",
            "elapsed_s": round(elapsed, 2),
            "error": err_msg,
        }

    before = to_float(summary.get("total_params_before"))
    after = to_float(summary.get("total_params_after"))
    if before > 0 and after > 0:
        param_ratio = round(after / before, 4)
    else:
        cr = to_float(summary.get("compression_rate"), 1.0)
        param_ratio = round(1.0 / cr, 4) if cr > 0 else 1.0

    actual_rank = summary.get("actual_rank") or summary.get("rank") or summary.get("tt_rank") or str(rank)

    return {
        "module": name,
        "model": model,
        "requested_rank": rank,
        "actual_rank": actual_rank,
        "status": summary.get("status", "UNKNOWN"),
        "accuracy_pct": to_float(summary.get("accuracy_pct")),
        "f1_score": to_float(summary.get("f1_score")),
        "num_compressed_layers": summary.get("num_compressed_layers", 0),
        "num_fallback_layers": summary.get("num_fallback_layers", 0),
        "total_params_before": summary.get("total_params_before", ""),
        "total_params_after": summary.get("total_params_after", ""),
        "param_ratio": param_ratio,
        "avg_reconstruction_error": summary.get("avg_reconstruction_error", ""),
        "elapsed_s": round(elapsed, 2),
        "error": summary.get("error", "") if str(summary.get("error")) != "nan" else "",
    }

def main():
    modules = load_manifest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    RESUME = "--resume" in sys.argv

    completed = set()
    all_results = []

    if RESUME and CSV_PATH.exists():
        try:
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    key = (r["module"], r["model"], int(r["requested_rank"]))
                    completed.add(key)
                    all_results.append(r)
            print(f"Resuming sweep: {len(completed)} configs already completed.")
        except Exception as e:
            print(f"Warning loading resume state: {e}")

    print(f"Rank Sweep starting at {datetime.now().isoformat()}")
    print(f"Modules: {len(modules)}")
    print(f"Ranks: {RANKS}")
    print(f"Models: {MODELS}")
    total_configs = len(modules) * len(RANKS) * len(MODELS)
    print(f"Total configs: {total_configs}")
    print()

    write_mode = "a" if (RESUME and CSV_PATH.exists() and len(completed) > 0) else "w"
    needs_header = not (RESUME and CSV_PATH.exists() and write_mode == "a")

    with open(CSV_PATH, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
            f.flush()

        for mod in modules:
            for model in MODELS:
                for rank in RANKS:
                    if RESUME and (mod["name"], model, rank) in completed:
                        print(f"Skipping completed: {mod['name']:20s} | {model:10s} | rank={rank}")
                        continue

                    label = f"[{mod['name']:25s} | {model:10s} | rank={rank:4d}]"
                    print(label, end=" ", flush=True)

                    row = run_one(mod, model, rank)
                    writer.writerow({k: row.get(k, "") for k in FIELDS})
                    f.flush()
                    all_results.append(row)

                    # Also update JSON periodically
                    with open(JSON_PATH, "w", encoding="utf-8") as jf:
                        json.dump(all_results, jf, indent=2)

                    status = str(row.get("status", "?"))
                    acc = str(row.get("accuracy_pct", "?"))
                    ratio = str(row.get("param_ratio", "?"))
                    comp = str(row.get("num_compressed_layers", "?"))
                    elapsed = str(row.get("elapsed_s", "?"))
                    print(f"{status:20s} acc={acc:>7} ratio={ratio:>7} comp={comp:>4} t={elapsed:>7}s")

    print()
    print(f"=== Sweep complete. Results written to {CSV_PATH} and {JSON_PATH}")

if __name__ == "__main__":
    main()
