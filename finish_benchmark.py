#!/usr/bin/env python3
"""
finish_benchmark.py — Master Orchestrator for remaining benchmark modules.

- Discovers all run_all_4_*.py that lack complete results
- Runs each with a hard 600s timeout using process-tree killing (taskkill /F /T)
- Direct log file streaming (no pipe deadlock on Windows)
- Uploads PASS modules to HuggingFace via huggingface_hub (no hf CLI)
- Persists state to finish_benchmark_state.json (crash-safe resume)
- Live logging every 30 seconds
- Final git push at end
"""

import os, sys, json, time, csv, logging, argparse, subprocess, threading
from pathlib import Path
from datetime import datetime, timedelta

REPO_DIR       = Path(__file__).resolve().parent
RESULTS_DIR    = REPO_DIR / "results"
STATE_FILE     = REPO_DIR / "finish_benchmark_state.json"
LOG_FILE       = REPO_DIR / "finish_benchmark.log"
MODELS         = ["distilbert", "roberta", "gpt2"]
PASS_THRESHOLD = 56.0
MODULE_TIMEOUT = 600
LOG_INTERVAL   = 30
HF_ORG         = "Hooshaai"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
LOG = logging.getLogger("orchestrator")

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

def extract_module_name(path):
    return path.stem[len("run_all_4_"):]

def is_module_done(module):
    unified = RESULTS_DIR / f"unified_{module}"
    if not unified.exists():
        return False
    for model in MODELS:
        csv_path = unified / model / "summary.csv"
        if not csv_path.exists():
            return False
    return True

def discover_remaining():
    return sorted([r for r in REPO_DIR.glob("run_all_4_*.py") if not is_module_done(extract_module_name(r))])

def read_summary_accuracy(module):
    results = {}
    unified = RESULTS_DIR / f"unified_{module}"
    for model in MODELS:
        csv_path = unified / model / "summary.csv"
        if csv_path.exists():
            try:
                with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    best_status, best_acc = "FAIL", 0.0
                    for row in reader:
                        try:
                            acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0)) or 0)
                            status = row.get("status", "FAIL")
                            if acc > best_acc:
                                best_acc, best_status = acc, status
                        except Exception:
                            pass
                results[model] = (best_status, best_acc)
            except Exception:
                pass
    return results

def upload_to_hf(module, model_slug, status, accuracy):
    try:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = f"{HF_ORG}/svd-linear-attention-{model_slug}-{module}"
        try:
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        except Exception:
            pass
        readme = f"---\ntags: [svd-linear-attention, {module}, pytorch]\ndatasets: [sst2]\n---\n# {module} / {model_slug}\nAccuracy: {accuracy:.2f}% | Status: {status}\n"
        api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id, commit_message=f"Update card acc={accuracy:.2f}%")
        unified = RESULTS_DIR / f"unified_{module}"
        if status == "PASS":
            weights = list((unified / model_slug).rglob("weights.pt"))
            if not weights:
                weights = list(unified.rglob("weights.pt"))
            if weights:
                LOG.info("Uploading weights.pt -> %s", repo_id)
                api.upload_file(path_or_fileobj=str(weights[0]), path_in_repo="weights.pt", repo_id=repo_id, commit_message="Upload weights")
        csv_path = unified / model_slug / "summary.csv"
        if csv_path.exists():
            api.upload_file(path_or_fileobj=str(csv_path), path_in_repo="summary.csv", repo_id=repo_id, commit_message="Upload metrics")
        LOG.info("HF upload done: %s", repo_id)
        return True
    except Exception as e:
        LOG.warning("HF upload failed %s/%s: %s", module, model_slug, e)
        return False

_current_module = ["(starting)"]
_completed = [0]
_total = [1]
_start_time = [time.time()]
_stop_log = threading.Event()

def _log_fn():
    while not _stop_log.wait(timeout=LOG_INTERVAL):
        elapsed = time.time() - _start_time[0]
        done, total = _completed[0], _total[0]
        avg = elapsed / max(done, 1)
        LOG.info("STATUS | Module: %s | Done: %d/%d | Elapsed: %s | ETA: %s",
                 _current_module[0], done, total,
                 str(timedelta(seconds=int(elapsed))),
                 str(timedelta(seconds=int((total - done) * avg))))

def kill_process_tree(pid):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=10)
        else:
            os.killpg(os.getpgid(pid), 9)
    except Exception:
        pass

def run_module(runner, module):
    env = os.environ.copy()
    env.update({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1",
                 "ORCHESTRATOR_UPLOAD": "0", "SKIP_HF_UPLOAD": "1"})
    cmd = [sys.executable, "-u", str(runner), "--models", *MODELS]
    log_path = REPO_DIR / "benchmark_logs" / f"finish_{module}.log"
    log_path.parent.mkdir(exist_ok=True)
    LOG.info("[%s] Launching runner: %s", module, runner.name)
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_DIR))
            try:
                proc.wait(timeout=MODULE_TIMEOUT)
                return proc.returncode
            except subprocess.TimeoutExpired:
                LOG.warning("[%s] TIMEOUT after %ds — terminating process tree PID %d.", module, MODULE_TIMEOUT, proc.pid)
                kill_process_tree(proc.pid)
                proc.wait(timeout=10)
                return -999
    except Exception as e:
        LOG.error("[%s] Exception: %s", module, e)
        return -1

def write_master_csv(state):
    out = RESULTS_DIR / "master_aggregated_results.csv"
    rows = [{"module": m, "exit_code": info.get("exit_code",""), "status": info.get("status",""),
              "distilbert_acc": info.get("accuracies", {}).get("distilbert", ["",0])[1],
              "roberta_acc": info.get("accuracies", {}).get("roberta", ["",0])[1],
              "gpt2_acc": info.get("accuracies", {}).get("gpt2", ["",0])[1],
              "elapsed_s": info.get("elapsed_s",""), "timestamp": info.get("timestamp","")}
             for m, info in state.items()]
    if not rows: return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    LOG.info("Master CSV written: %s", out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    state = load_state()
    remaining = discover_remaining()
    done_statuses = ("PASS", "PASS_LOW", "FAIL", "FAILED_LOW", "TIMEOUT", "CRASH")
    to_run = [r for r in remaining if extract_module_name(r) not in state or state[extract_module_name(r)].get("status") not in done_statuses]

    LOG.info("Remaining (no results): %d | To run this session: %d", len(remaining), len(to_run))
    if args.dry_run:
        [print(f"  {extract_module_name(r)}") for r in to_run]; return

    _total[0] = len(to_run); _start_time[0] = time.time()
    threading.Thread(target=_log_fn, daemon=True).start()

    pass_c = pass_low_c = fail_c = timeout_c = 0

    for idx, runner in enumerate(to_run):
        module = extract_module_name(runner)
        _current_module[0] = module
        LOG.info("=" * 60)
        LOG.info("[%d/%d] Running: %s", idx + 1, len(to_run), module)
        t0 = time.time()
        exit_code = run_module(runner, module)
        elapsed = time.time() - t0
        accs = read_summary_accuracy(module)
        LOG.info("[%s] Accuracies: %s | exit=%d | elapsed=%.1fs", module, accs, exit_code, elapsed)

        if exit_code == -999:
            overall = "TIMEOUT"; timeout_c += 1
        elif exit_code != 0:
            overall = "CRASH"; fail_c += 1
        else:
            any_pass = any(s == "PASS" for s, _ in accs.values())
            any_low  = any(s == "PASS_LOW" for s, _ in accs.values())
            any_ok_acc = any(acc >= PASS_THRESHOLD for _, acc in accs.values())
            if any_pass or (any_ok_acc and not any_low):
                overall = "PASS"; pass_c += 1
            elif any_low or any_ok_acc:
                overall = "PASS_LOW"; pass_low_c += 1
            else:
                overall = "FAILED_LOW"; fail_c += 1

        LOG.info("[%s] Overall: %s", module, overall)
        if not args.no_upload and overall in ("PASS", "PASS_LOW"):
            for model_slug, (model_status, model_acc) in accs.items():
                if model_acc >= PASS_THRESHOLD:
                    upload_to_hf(module, model_slug, model_status, model_acc)

        state[module] = {"status": overall, "exit_code": exit_code, "elapsed_s": round(elapsed, 1),
                          "accuracies": {k: list(v) for k, v in accs.items()}, "timestamp": datetime.now().isoformat()}
        save_state(state)
        _completed[0] = idx + 1

    _stop_log.set()
    write_master_csv(state)
    total_time = time.time() - _start_time[0]
    LOG.info("=" * 60)
    LOG.info("FINAL | PASS=%d | PASS_LOW=%d | FAIL/CRASH=%d | TIMEOUT=%d | Time=%s",
             pass_c, pass_low_c, fail_c, timeout_c, str(timedelta(seconds=int(total_time))))

    LOG.info("Git push...")
    try:
        subprocess.run(["git", "add", "-A"], cwd=REPO_DIR, check=True)
        subprocess.run(["git", "commit", "-m", "chore: finalize remaining benchmark modules"], cwd=REPO_DIR)
        subprocess.run(["git", "push", "origin", "main"], cwd=REPO_DIR,
                       env={**os.environ, "GIT_SSH_COMMAND": "ssh -o Port=443 -o HostName=ssh.github.com"})
    except Exception as e:
        LOG.warning("Git push failed: %s", e)
    LOG.info("Done!")

if __name__ == "__main__":
    main()
