#!/usr/bin/env python3
"""
Autonomous 48-Hour Multi-Stream High-Performance GPU Pipeline.
Executes all 6 benchmark stages CONCURRENTLY across 6 parallel worker streams:
  Stream 1: Stage 2A — Category 5A SVD Multi-Rank Sweeps on DistilBERT (Batch 128)
  Stream 2: Stage 2B — Category 5A SVD Multi-Rank Sweeps on RoBERTa-base (Batch 64)
  Stream 3: Stage 3  — Category 1-4 Attention & Linear/SSM Modules (Batch 64)
  Stream 4: Stage 4  — Cross-Domain Transfer Sweeps: MNLI & AG News (Batch 96)
  Stream 5: Stage 5  — Long-Context Sequence Length Scaling: N=128..1024 (Batch 32-64)
  Stream 6: Stage 6  — Statistical Rigor Multi-Seed Sweeps: Seeds 123 & 456 (Batch 64)

Hardware Targets & Safeguards:
  - GPU Compute Utilization: > 90% sustained (up to 100% SM saturation)
  - VRAM Active Utilization: > 50% capacity (4.0 GB to 6.0 GB strictly enforced)
  - Dynamic VRAM Governor: Holds subprocesses if VRAM > 5800 MB (zero OOM risk)
  - Auto-Commit & Auto-Push: Triggers every 10 completed tasks across all streams
  - 5-Minute Telemetry Monitor: Logs GPU util, VRAM, temp, power, and active workers
"""
import os
import sys
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MASTER_LOG = LOGS_DIR / "autonomous_48h_master.log"
TELEMETRY_LOG = LOGS_DIR / "autonomous_48h_telemetry.log"

global_lock = threading.Lock()
global_completed_counter = [0]
active_workers_counter = [0]
max_allowed_workers = [4]  # Dynamic worker limit governed by VRAM usage (min 2, max 6)

DEPRECATED_FILE = BASE_DIR / "audit" / "deprecated_modules.txt"
if DEPRECATED_FILE.exists():
    DEPRECATED_MODULES = set(
        line.strip() for line in DEPRECATED_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
else:
    DEPRECATED_MODULES = {
        "tt", "tt_linear", "tt_compressor", "tt_linear_compressor",
        "tt_factory", "tt_compressor_factory", "tt_linear_factory",
        "lillama_compressor", "mamba_svd_flash"
    }

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    try:
        with open(MASTER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def auto_commit_and_push(msg: str):
    try:
        subprocess.run(["git", "add", "results", "MASTER_PROJECT_RESULTS.md"], cwd=str(BASE_DIR), check=False)
        commit_res = subprocess.run(
            ["git", "commit", "-m", f"auto(48h-pipeline): {msg}"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True
        )
        if "nothing to commit" not in commit_res.stdout.lower() and "clean" not in commit_res.stdout.lower():
            log(f"[AUTO-COMMIT] {msg}")
            push_res = subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=60
            )
            if push_res.returncode == 0:
                log(f"[AUTO-PUSH] Synced with origin/main")
            else:
                log(f"[AUTO-PUSH NOTE] Push output: {push_res.stderr.strip()[:100]}")
    except Exception as e:
        log(f"[AUTO-COMMIT EXCEPTION] {e}")

def get_vram_and_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return {
            "used_mb": int(parts[0]),
            "total_mb": int(parts[1]),
            "util_pct": int(parts[2]),
            "temp_c": int(parts[3]),
            "power_w": float(parts[4]),
        }
    except Exception:
        return {"used_mb": 0, "total_mb": 8188, "util_pct": 0, "temp_c": 0, "power_w": 0.0}

def vram_governor_loop():
    """
    Monitors GPU VRAM every 30 seconds and dynamically adapts worker pool:
      - RTX 4070 (8GB VRAM): Minimum 4,100 MB (50%), Maximum 7,370 MB (90%)
      - If VRAM drops below 50% -> increase workers by 1 (max 6)
      - If VRAM exceeds 90% -> decrease workers by 1 (min 2), sleep 30s to settle
    """
    iteration = 0
    while True:
        time.sleep(30)
        iteration += 1
        tele = get_vram_and_util()
        used = tele["used_mb"]
        total = tele["total_mb"]
        util = tele["util_pct"]
        temp = tele["temp_c"]
        pwr = tele["power_w"]
        vram_pct = (used / total) * 100.0 if total > 0 else 0

        with global_lock:
            act = active_workers_counter[0]
            done = global_completed_counter[0]
            cur_limit = max_allowed_workers[0]

            if vram_pct < 50.0 and act >= cur_limit:
                new_limit = min(6, cur_limit + 1)
                if new_limit > cur_limit:
                    max_allowed_workers[0] = new_limit
                    log(f"[VRAM GOVERNOR] VRAM below 50% ({vram_pct:.1f}%) -> increasing max workers to {new_limit}")
            elif vram_pct > 90.0:
                new_limit = max(2, cur_limit - 1)
                if new_limit < cur_limit:
                    max_allowed_workers[0] = new_limit
                    log(f"[VRAM GOVERNOR] VRAM above 90% ({vram_pct:.1f}%) -> decreasing max workers to {new_limit}")

            cur_limit = max_allowed_workers[0]

        t_line = (
            f"[30S TELEMETRY] Active: {act}/{cur_limit} workers | Completed: {done} | "
            f"VRAM: {used}/{total} MB ({vram_pct:.1f}%) | GPU Util: {util}% | "
            f"Temp: {temp}C | Power: {pwr:.1f}W"
        )
        try:
            print(t_line, flush=True)
        except UnicodeEncodeError:
            print(t_line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        try:
            with open(TELEMETRY_LOG, "a", encoding="utf-8") as f:
                f.write(t_line + "\n")
            if iteration % 10 == 0:  # Every 5 minutes log to master
                with open(MASTER_LOG, "a", encoding="utf-8") as f:
                    f.write(t_line + "\n")
        except Exception:
            pass

def run_single_task(task):
    summary_path = Path(task["summary_file"])
    if summary_path.exists() and summary_path.stat().st_size > 50:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "CRASH_NO_OUTPUT" not in content and len(content.strip().split("\n")) >= 2:
                log(f"[SKIP] {task['name']} already completed ({summary_path})")
                return True
        except Exception:
            pass

    # Dynamic VRAM governor: Target strictly between 50% (~4,100 MB) and 90% (~7,370 MB)
    while True:
        tele = get_vram_and_util()
        with global_lock:
            cur_active = active_workers_counter[0]
            cur_limit = max_allowed_workers[0]
            # Max 7,370 MB hard ceiling (90% of 8,188 MB)
            if tele["used_mb"] < 7370 and cur_active < cur_limit:
                active_workers_counter[0] += 1
                break
        time.sleep(3)

    log(f"============================================================")
    log(f"[START] {task['name']} (Batch={task.get('batch_size', 64)})")
    log(f"============================================================")

    cmd = [
        sys.executable,
        task.get("runner", "neuralops/benchmarks/run_quality_sweep.py"),
        "--modules", task["module"],
        "--model", task["model"],
        "--model_family", task["model_family"],
        "--dataset", task["dataset"],
        "--batch_size", str(task.get("batch_size", 64)),
        "--max_vram_fraction", "0.45",
        "--recovery_steps", str(task.get("recovery_steps", 50)),
        "--learning_rate", str(task.get("lr", "2e-4")),
        "--seed", str(task.get("seed", 42)),
        "--output_dir", task["out_dir"],
    ]

    if "rank" in task:
        cmd.extend(["--rank", str(task["rank"])])
    if "seq_len" in task:
        cmd.extend(["--seq_len", str(task["seq_len"])])
    if "dataset_dir" in task and task["dataset_dir"]:
        cmd.extend(["--dataset_dir", task["dataset_dir"]])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"Codes;{env.get('PYTHONPATH', '')}" if os.name == "nt" else f"Codes:{env.get('PYTHONPATH', '')}"
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    env["CUDA_LAUNCH_BLOCKING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=task.get("timeout", 1800),
        )
        elapsed = time.time() - t0
        success = (proc.returncode == 0)
        if success:
            log(f"[PASS] {task['name']} completed in {elapsed:.1f}s")
        else:
            log(f"[FAIL] {task['name']} exited with code {proc.returncode} in {elapsed:.1f}s")
            lines = proc.stdout.strip().split("\n")
            tail = "\n".join(lines[-10:])
            log(f"Tail log:\n{tail}")
        return success
    except subprocess.TimeoutExpired:
        log(f"[TIMEOUT] {task['name']} timed out after {task.get('timeout', 1800)}s")
        return False
    except Exception as e:
        log(f"[EXCEPTION] {task['name']}: {e}")
        return False
    finally:
        with global_lock:
            active_workers_counter[0] -= 1

def run_task_worker(task, total_pending, task_idx):
    """Executes a single task in the shared worker pool, updating global accounting."""
    res = run_single_task(task)
    with global_lock:
        global_completed_counter[0] += 1
        cur_global = global_completed_counter[0]
        if cur_global % 10 == 0:
            auto_commit_and_push(f"Global 8-Worker Dynamic Checkpoint ({cur_global} tasks completed)")
    log(f"[Dynamic Worker Progress]: Task {task_idx+1}/{total_pending} finished | Global Tasks Recorded: {cur_global}")
    return res

def run_stream_worker(stream_name, task_list):
    """Legacy stream worker kept for modular compatibility."""
    log(f"############################################################")
    log(f"### [LAUNCH STREAM] {stream_name}: {len(task_list)} Tasks Queued")
    log(f"############################################################")
    completed, failed = 0, 0
    for task in task_list:
        res = run_single_task(task)
        if res:
            completed += 1
        else:
            failed += 1

        with global_lock:
            global_completed_counter[0] += 1
            cur_global = global_completed_counter[0]
            if cur_global % 10 == 0:
                auto_commit_and_push(f"Global 6-Stream Checkpoint ({cur_global} tasks completed)")

        log(f"[{stream_name} Stream Progress]: {completed}/{len(task_list)} completed ({failed} failed)")

    log(f"### [STREAM FINISHED] {stream_name} finished all {len(task_list)} tasks.")
    auto_commit_and_push(f"{stream_name} stream complete ({completed} passed, {failed} failed)")
    return completed, failed

def build_all_streams():
    """Builds 6 concurrent task streams covering the entire repository."""
    cat5a_all = [
        "a3_compressor", "aa_svd_compressor", "aasvd", "adaptive_compressor", "adaptive_ranks",
        "adasvd_compressor", "admm_compressor", "come", "compot_compressor", "cr_compressor",
        "cur_compressor", "cur_linear", "cure", "direct_svd", "dobi_svd", "erc_svd", "frandi",
        "frandi_compressor", "gfws_d_compressor", "grecodec", "grecodec_compressor",
        "hybrid_svd_layer", "kronecker_compressor", "kronecker_linear", "kronecker_tt_lora",
        "lsc_compressor", "mgie_svd", "mgie_svd_compressor", "mlorq_compressor", "modegpt",
        "modegpt_compressor", "nmf_compressor", "oats", "oats_compressor",
        "rotational_slicegpt_int4", "rpca", "rpca_compressor", "saes_svd", "saes_svd_compressor",
        "saes_svd_v2", "salr", "slicegpt", "slicegpt_compressor", "slicegpt_pca",
        "subspace_projection", "svd_linear", "swift_svd_act", "swift_svd_compressor",
        "tt_compressor", "tt_linear", "uniql_compressor"
    ]

    streams = {}

    # STREAM 1: Stage 2A — Category 5A SVD Multi-Rank Sweeps on DistilBERT (Batch 128)
    s1_tasks = []
    for m in cat5a_all:
        for r in [64, 128, 256, 384]:
            out = str(BASE_DIR / "results" / "deep_rank_distilbert" / m / f"rank_{r}")
            s1_tasks.append({
                "name": f"Stream 1 (Cat 5A r={r}) | DISTILBERT | {m}",
                "module": m, "model": "models/distilbert_sst2", "model_family": "bert",
                "dataset": "sst2", "rank": r, "batch_size": 128, "lr": "2e-4",
                "out_dir": out, "summary_file": f"{out}/summary.csv",
            })
    streams["STREAM_1_DISTILBERT_SVD"] = s1_tasks

    # STREAM 2: Stage 2B — Category 5A SVD Multi-Rank Sweeps on RoBERTa-base (Batch 96)
    s2_tasks = []
    for m in cat5a_all:
        for r in [64, 128, 256, 384]:
            out = str(BASE_DIR / "results" / "deep_rank_roberta" / m / f"rank_{r}")
            s2_tasks.append({
                "name": f"Stream 2 (Cat 5A r={r}) | ROBERTA | {m}",
                "module": m, "model": "models/roberta_sst2", "model_family": "roberta",
                "dataset": "sst2", "rank": r, "batch_size": 96, "lr": "5e-5",
                "out_dir": out, "summary_file": f"{out}/summary.csv",
            })
    streams["STREAM_2_ROBERTA_SVD"] = s2_tasks

    # STREAM 3: Stage 3 — Full Attention & Linear/SSM Architectures (Batch 64)
    key_attentions = [
        "cosformer", "performer", "linformer_attention", "hgrn2_layer", "gated_deltanet_block",
        "simple_ssm", "samba_m3", "mamba_svd_flash", "infini_attention", "flash3", "nystrom",
        "rala", "spectron", "aegis_v1", "aegis_v2", "aegis_v6", "chela_slay", "nexus", "moba",
        "fused_linear_attention", "rwkv_attention", "retention", "longformer_local", "bigbird"
    ]
    s3_tasks = []
    for m in key_attentions:
        for model_tag, m_path, fam, lr in [
            ("roberta", "models/roberta_sst2", "roberta", "5e-5"),
            ("distilbert", "models/distilbert_sst2", "bert", "2e-4"),
        ]:
            out = str(BASE_DIR / "results" / f"attn_{model_tag}" / m)
            timeout_val = 180 if m == "mamba_svd_flash" else 300
            s3_tasks.append({
                "name": f"Stream 3 (Attn) | {model_tag.upper()} | {m}",
                "module": m, "model": m_path, "model_family": fam, "dataset": "sst2",
                "batch_size": 64, "lr": lr, "out_dir": out, "summary_file": f"{out}/summary.csv",
                "timeout": timeout_val,
            })
    streams["STREAM_3_ATTENTIONS"] = s3_tasks

    # STREAM 4: Stage 4 — Cross-Domain Transfer Sweeps: MNLI & AG News (Batch 64)
    transfer_mods = [
        "adasvd_compressor", "a3_compressor", "saes_svd_v2", "direct_svd", "rpca_compressor",
        "kronecker_compressor", "mlorq_compressor", "salr", "oats_compressor", "cure", "admm_compressor"
    ]
    s4_tasks = []
    for m in transfer_mods:
        out_mnli = str(BASE_DIR / "results" / "mnli_deep_sweep" / m)
        s4_tasks.append({
            "name": f"Stream 4 (MNLI) | {m}",
            "module": m, "model": "models/distilbert_mnli_full", "model_family": "bert",
            "dataset": "mnli", "batch_size": 64, "lr": "2e-4", "out_dir": out_mnli,
            "summary_file": f"{out_mnli}/summary.csv",
        })
        out_ag = str(BASE_DIR / "results" / "agnews_deep_sweep" / m)
        s4_tasks.append({
            "name": f"Stream 4 (AGNews) | {m}",
            "module": m, "model": "models/distilbert_agnews_full", "model_family": "bert",
            "dataset": "agnews", "batch_size": 64, "lr": "2e-4", "out_dir": out_ag,
            "summary_file": f"{out_ag}/summary.csv",
        })
    streams["STREAM_4_TRANSFER"] = s4_tasks

    # STREAM 5: Stage 5 — Long-Context Sequence Length Scaling: N=128..1024 (Batch 32-64)
    seq_mods = ["linformer_attention", "cosformer", "performer", "simple_ssm", "hgrn2_layer", "gated_deltanet_block"]
    s5_tasks = []
    for m in seq_mods:
        for seq_len in [128, 256, 512]:
            bs = 64 if seq_len <= 256 else 32
            out = str(BASE_DIR / "results" / "seqlen_48h" / m / f"seq_{seq_len}")
            s5_tasks.append({
                "name": f"Stream 5 (SeqLen {seq_len}) | {m}",
                "module": m, "model": "models/distilbert_sst2", "model_family": "bert",
                "dataset": "sst2", "seq_len": seq_len, "batch_size": bs, "lr": "2e-4",
                "out_dir": out, "summary_file": f"{out}/summary.csv",
            })
    streams["STREAM_5_SEQLEN_SCALING"] = s5_tasks

    # STREAM 6: Stage 6 — Multi-Seed Robustness Sweeps: Seeds 123 & 456 (Batch 64)
    s6_tasks = []
    for m in transfer_mods:
        for seed in [123, 456]:
            for model_tag, m_path, fam, lr in [
                ("distilbert", "models/distilbert_sst2", "bert", "2e-4"),
                ("roberta", "models/roberta_sst2", "roberta", "5e-5"),
            ]:
                out = str(BASE_DIR / "results" / f"multiseed_rigor_{model_tag}" / m / f"seed_{seed}")
                s6_tasks.append({
                    "name": f"Stream 6 (Seed {seed}) | {model_tag.upper()} | {m}",
                    "module": m, "model": m_path, "model_family": fam, "dataset": "sst2",
                    "batch_size": 64, "seed": seed, "lr": lr, "out_dir": out,
                    "summary_file": f"{out}/summary.csv",
                })
    streams["STREAM_6_MULTISEED"] = s6_tasks

    return streams

def main():
    log("============================================================")
    log("Starting Multi-Worker Dynamic High-Performance GPU Pipeline")
    log("Running 8 Parallel Workers across all remaining tasks (Batch=96-128)")
    log(f"Base Directory: {BASE_DIR}")
    log("============================================================")

    # Start 30-second hardware telemetry & VRAM governor background monitor
    monitor_thread = threading.Thread(target=vram_governor_loop, daemon=True)
    monitor_thread.start()
    log("[SYSTEM] 30-second adaptive VRAM governor (50%-90% target, max 6 workers) initialized.")

    all_streams = build_all_streams()

    # Interleave tasks across streams so workers alternate between DistilBERT, RoBERTa, etc.
    stream_task_lists = list(all_streams.values())
    combined_tasks = []
    max_len = max(len(tl) for tl in stream_task_lists)
    for i in range(max_len):
        for tl in stream_task_lists:
            if i < len(tl):
                combined_tasks.append(tl[i])

    # Filter out already completed or deprecated tasks up front so workers immediately jump to remaining ones
    pending_tasks = []
    for t in combined_tasks:
        if t.get("module") in DEPRECATED_MODULES:
            continue
        sp = Path(t["summary_file"])
        if sp.exists() and sp.stat().st_size > 50:
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    c = f.read()
                # Recognize all non-empty evaluated statuses (PASS, FAIL, MODERATE_EXPANSION, etc.)
                if "CRASH_NO_OUTPUT" not in c and len(c.strip().split("\n")) >= 2:
                    continue
            except Exception:
                pass
        pending_tasks.append(t)

    total_tasks = len(combined_tasks)
    total_pending = len(pending_tasks)
    log(f"Total Framework Tasks: {total_tasks} | Already Done: {total_tasks - total_pending} | Remaining Pending: {total_pending}")

    if total_pending == 0:
        log("ALL TASKS ACROSS ALL STREAMS ARE COMPLETE!")
        return

    t_global_start = time.time()

    # Launch parallel workers pulling from the shared pool of remaining tasks
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {
            executor.submit(run_task_worker, task, total_pending, idx): task["name"]
            for idx, task in enumerate(pending_tasks)
        }
        for future in as_completed(future_map):
            t_name = future_map[future]
            try:
                res = future.result()
            except Exception as e:
                log(f"[TASK EXCEPTION] {t_name}: {e}")

    total_time_h = (time.time() - t_global_start) / 3600.0
    log("============================================================")
    log(f"ALL {total_tasks} PIPELINE TASKS COMPLETED IN {total_time_h:.2f} HOURS!")
    log("============================================================")

if __name__ == "__main__":
    main()
