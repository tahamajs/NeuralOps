#!/usr/bin/env python3
"""
True Multi-Process GPU Parallel Orchestrator (Bypasses Python GIL)
Spawns 3 independent OS-level Python processes concurrently on CUDA.
Pins GPU Utilization to 90-100% and saturates RTX 4070 VRAM (~6.5-7.5 GB / 8 GB).
Live Telemetry + Per-Run Git Auto-Commit & Push.
"""
import os
import sys
import gc
import json
import time
import logging
import multiprocessing as mp
from pathlib import Path

# Force 100% Offline Mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MASTER_LOG = PROJECT_DIR / "trajectory_master.log"
RESULTS_DIR = BASE_DIR / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WORKER_STATE_FILE = BASE_DIR / "worker_status.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MultiProcess_GPU_Orchestrator")


def log_master(msg: str):
    logger.info(msg)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(MASTER_LOG, "a", encoding="utf-8", errors="ignore") as f:
            f.write(line)
    except Exception:
        pass


def update_worker_status_file(worker_id: int, status_dict: dict):
    try:
        worker_file = BASE_DIR / f"worker_status_{worker_id}.json"
        with open(worker_file, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, indent=2)
    except Exception:
        pass


def get_completed_runs():
    completed = set()
    if MASTER_LOG.exists():
        try:
            with open(MASTER_LOG, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "<< SUCCESS:" in line or "NOTICE:" in line:
                        if "attn=" in line and "task=" in line:
                            parts = line.split(" | ")
                            a = [p for p in parts if "attn=" in p]
                            t = [p for p in parts if "task=" in p]
                            if a and t:
                                attn_val = a[0].split("=")[1].strip()
                                task_val = t[0].split("=")[1].strip()
                                completed.add((attn_val, task_val))
        except Exception:
            pass
    return completed


def worker_process_main(worker_id: int, task_queue: mp.Queue, total_tasks: int, git_lock: mp.Lock):
    # Separate Python process running on its own CPU core & CUDA context
    import torch
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        # Hard cap at 40% of 8GB = 3.2GB VRAM per worker (2 workers = 6.4GB max) to perfectly prevent Windows Shared Memory swapping
        torch.cuda.set_per_process_memory_fraction(0.40)
    from main import run_train, _import_framework
    fw = _import_framework()

    logger.info(f"Process Worker-{worker_id} (PID {os.getpid()}) started on CUDA.")
    update_worker_status_file(worker_id, {
        "status": "idle", "attn": "-", "task": "-", "pid": os.getpid(), "idx": "--"
    })

    while True:
        try:
            item = task_queue.get(timeout=2)
        except Exception:
            break

        attn, task, idx = item
        completed_runs = get_completed_runs()
        if (attn, task) in completed_runs:
            log_master(f"[Worker-{worker_id}] [{idx}/{total_tasks}] -- SKIPPING: attn={attn} | task={task} (Already completed)")
            continue

        log_master(f"[Worker-{worker_id}] [{idx}/{total_tasks}] >> Launching: attn={attn} | task={task} | seed=42")
        update_worker_status_file(worker_id, {
            "status": "running", "attn": attn, "task": task, "idx": f"{idx}/{total_tasks}", "pid": os.getpid(), "start_time": time.time()
        })

        cfg = fw["ExperimentConfig"]()
        cfg.attn_type = attn
        cfg.task = task
        cfg.epochs = 3.0               # Increased from 1.0 to 3.0 to heavily improve training results (accuracy) on 1000 samples
        cfg.max_train_samples = 1000   # Reverted to 1000 samples for ultra-fast benchmarking

        cfg.output_dir = str(RESULTS_DIR)
        cfg.save_model = False
        cfg.seed = 42

        # Maximize GPU utilization on RTX 4070 8GB (single worker)
        heavy_attn = {"full", "flash3", "griffin_ssd", "aegis_v3", "spectra_v2",
                      "samba_m3", "mamba", "gated_delta_net", "spectral_mamba",
                      "aegis_hybrid", "mamba_svd_flash", "mamba_inject",
                      "spectron", "hgrn2_layer", "simple_ssm", "samba_moe",
                      "primal", "projection_factorized_attn", "qcqa",
                      "mamba_svd_flash_inject",
                      "mla", "moba", "moe_deltakv_paged", "nasc",
                      "nasc_attention", "nasc_injection", "nexus", "nha"}
        super_heavy_attn = {"factorised_matrix_state", "hgrn2_outer", "kda", "leka_v1", "leka_v2", "leka_v3"}
        extreme_attn = {"expert_choice_moe", "granite_hybrid", "jamba2", "inject_aegis_spectron_fp8"}
                            
        if attn in extreme_attn:
            cfg.batch_size = 4
        elif attn in super_heavy_attn:
            cfg.batch_size = 8
        elif attn in heavy_attn:
            cfg.batch_size = 32
        elif attn in {"adasplash", "nha_compressor", "token_compressed"}:
            cfg.batch_size = 16
        else:
            cfg.batch_size = 64   # Optimal batch size for 2 workers

        # Aggressive learning rate for fast convergence on small dataset
        cfg.lr = 1e-4 if attn in ("svd_linear", "full") else 2e-4

        # Dynamic Override from UI Settings Panel
        settings_file = PROJECT_DIR / "orchestrator_settings.json"
        if settings_file.exists():
            try:
                import json
                user_settings = json.loads(settings_file.read_text(encoding="utf-8"))
                if "batch_size" in user_settings:
                    cfg.batch_size = user_settings["batch_size"]
                if "dl_workers" in user_settings:
                    cfg.num_workers = user_settings["dl_workers"]
                if "lr" in user_settings and user_settings["lr"]:
                    cfg.lr = user_settings["lr"]
                if "epochs" in user_settings and user_settings["epochs"]:
                    cfg.epochs = user_settings["epochs"]
                if "seed" in user_settings and user_settings["seed"]:
                    cfg.seed = user_settings["seed"]
            except Exception:
                pass

        t0 = time.time()
        acc, f1, loss = 0.0, 0.0, 0.0
        try:
            # Save the training configuration to disk so the telemetry thread can capture it
            exp_dir = RESULTS_DIR / f"{attn}_{task}_s{cfg.seed}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_dir / "run_config.json", "w", encoding="utf-8") as f:
                import json
                json.dump({
                    "batch_size": cfg.batch_size,
                    "num_workers": getattr(cfg, "num_workers", 4),
                    "learning_rate": cfg.lr,
                    "epochs": cfg.epochs,
                    "seed": cfg.seed,
                    "attn": attn,
                    "task": task
                }, f)

            result = run_train(cfg, fw, logger)
            elapsed = time.time() - t0
            eval_metrics = result.get("eval", {})
            acc = eval_metrics.get("eval_accuracy", 0.0)
            f1 = eval_metrics.get("eval_f1_macro", 0.0)
            loss = eval_metrics.get("eval_loss", 0.0)
            
            # Save final metrics
            with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"accuracy": round(acc*100, 2), "f1": round(f1*100, 2), "loss": round(loss, 4), "time_seconds": round(elapsed, 1)}, f)

            log_master(f"[Worker-{worker_id}] [{idx}/{total_tasks}] << SUCCESS: attn={attn} | task={task} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Loss: {loss:.4f} | Time: {elapsed:.1f}s")
            update_worker_status_file(worker_id, {
                "status": "completed", "attn": attn, "task": task, "idx": f"{idx}/{total_tasks}", "last_acc": round(acc*100, 2), "last_loss": round(loss, 4), "pid": os.getpid()
            })
            commit_msg = f"feat(benchmark): [{idx}/{total_tasks}] {attn} on {task} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Loss: {loss:.4f}"
        except Exception as e:
            elapsed = time.time() - t0
            log_master(f"[Worker-{worker_id}] [{idx}/{total_tasks}] NOTICE: attn={attn} | task={task} | Error: {e} | Time: {elapsed:.1f}s")
            update_worker_status_file(worker_id, {
                "status": "error", "attn": attn, "task": task, "idx": f"{idx}/{total_tasks}", "error": str(e), "pid": os.getpid()
            })
            commit_msg = f"fix(crash): [{idx}/{total_tasks}] {attn} on {task} failed with exception"

        # Auto-commit to Git
        try:
            import subprocess, glob
            # Generate the unified history json for the Docs frontend
            hist_res = {}
            for p in glob.glob(str(RESULTS_DIR / "*" / "history" / "log_history.json")):
                d = os.path.basename(os.path.dirname(os.path.dirname(p)))
                try: hist_res[d] = json.load(open(p, encoding="utf-8", errors="ignore"))
                except: pass
            
            docs_res_dir = PROJECT_DIR / "Docs" / "results"
            docs_res_dir.mkdir(parents=True, exist_ok=True)
            with open(docs_res_dir / "all_histories.json", "w", encoding="utf-8") as f:
                json.dump(hist_res, f)

            with git_lock:
                subprocess.run(["git", "add", "-f", str(RESULTS_DIR), str(MASTER_LOG), str(docs_res_dir / "all_histories.json")], cwd=str(PROJECT_DIR), check=True)
                subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(PROJECT_DIR), capture_output=True, text=True, check=False)
                subprocess.run(["git", "pull", "origin", "main", "--rebase"], cwd=str(PROJECT_DIR), capture_output=True, check=False)
                subprocess.run(["git", "push", "origin", "main"], cwd=str(PROJECT_DIR), capture_output=True, text=True, check=False)
        except Exception as e:
            logger.warning(f"Git push error: {e}")

        # Clean GPU VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    update_worker_status_file(worker_id, {"status": "finished", "attn": "-", "task": "-", "pid": os.getpid()})
    logger.info(f"Worker-{worker_id} finished all tasks.")


def telemetry_sync_thread():
    import time, subprocess, glob, json, os
    from pathlib import Path
    docs_res_dir = PROJECT_DIR / "Docs" / "results"
    while True:
        try:
            # Combine worker statuses
            combined_status = {}
            for i in range(1, 10):
                w_file = BASE_DIR / f"worker_status_{i}.json"
                if w_file.exists():
                    try:
                        with open(w_file, "r", encoding="utf-8") as f:
                            combined_status[str(i)] = json.load(f)
                    except Exception:
                        pass
            if combined_status:
                try:
                    with open(BASE_DIR / "worker_status.json", "w", encoding="utf-8") as f:
                        json.dump(combined_status, f, indent=2)
                except Exception:
                    pass

            hist_res = {}
            for p in glob.glob(str(RESULTS_DIR / "*" / "history" / "log_history.json")):
                d = os.path.basename(os.path.dirname(os.path.dirname(p)))
                try: 
                    hist_data = json.load(open(p, encoding="utf-8", errors="ignore"))
                    
                    config_data = {}
                    config_path = os.path.join(os.path.dirname(os.path.dirname(p)), "run_config.json")
                    if os.path.exists(config_path):
                        config_data = json.load(open(config_path, encoding="utf-8", errors="ignore"))
                        
                    metrics_data = {}
                    metrics_path = os.path.join(os.path.dirname(os.path.dirname(p)), "metrics.json")
                    if os.path.exists(metrics_path):
                        metrics_data = json.load(open(metrics_path, encoding="utf-8", errors="ignore"))

                    hist_res[d] = {
                        "history": hist_data,
                        "config": config_data,
                        "metrics": metrics_data
                    }
                except: pass
            
            docs_res_dir.mkdir(parents=True, exist_ok=True)
            out_file = docs_res_dir / "all_histories.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(hist_res, f)

            # Check if there are changes before pushing to avoid empty commits
            status = subprocess.run(["git", "status", "--porcelain", str(out_file)], cwd=str(PROJECT_DIR), capture_output=True, text=True)
            if status.stdout.strip():
                subprocess.run(["git", "add", "-f", str(out_file)], cwd=str(PROJECT_DIR), check=False)
                subprocess.run(["git", "commit", "-m", "chore(telemetry): Live training telemetry update"], cwd=str(PROJECT_DIR), capture_output=True, check=False)
                subprocess.run(["git", "pull", "origin", "main", "--rebase"], cwd=str(PROJECT_DIR), capture_output=True, check=False)
                subprocess.run(["git", "push", "origin", "main"], cwd=str(PROJECT_DIR), capture_output=True, check=False)
        except Exception as e:
            pass
        time.sleep(60)

def main():
    # Set multiprocessing start method for Windows
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    log_master("=" * 60)
    log_master("  STARTING MULTI-PROCESS TRIPLE-WORKER GPU ORCHESTRATION  ")
    log_master("=" * 60)

    import threading
    t = threading.Thread(target=telemetry_sync_thread, daemon=True)
    t.start()
    log_master("Started background telemetry sync thread.")

    ATTN_MODELS = [
        "adaptive_span",
        "adaptive_svd",
        "adaptive_svd_attn",
        "adasplash",
        "aegis_hybrid",
        "aegis_hybrid_attn",
        "aegis_spectron_fp8",
        "aegis_spectron_fp8_attn",
        "aegis_v2",
        "aegis_v3",
        "aegis_v4",
        "aegis_v5",
        "aegis_v6",
        "asvd",
        "bam_umoe",
        "block_transformer_hiba",
        "chela_slay",
        "cluster",
        "compressed_attention_layer",
        "compressed_attention_layer_energy",
        "compressed_mha",
        "compression_unified_rope",
        "decay_linear_attn",
        "eat",
        "expert_choice_moe",
        "factorised_matrix_state",
        "flash3",
        "flash3_fp8",
        "full",
        "fusion_attn",
        "galerkin",
        "gated_delta_net",
        "gated_delta_net_inject",
        "gated_deltanet_block",
        "gated_linear_attn",
        "goldfinch_ahn",
        "gqa",
        "granite_hybrid",
        "griffin_ssd",
        "halo_regla",
        "hamming",
        "hgrn2_gated_deltanet",
        "hgrn2_layer",
        "hgrn2_outer",
        "hla",
        "hymba_transx",
        "infini",
        "inject_aegis_hybrid",
        "inject_aegis_spectron_fp8",
        "jamba2",
        "jamba_info",
        "kda",
        "leka_v1",
        "leka_v2",
        "leka_v3",
        "linformer",
        "lola_attn",
        "lora_attn",
        "lsh",
        "mamba",
        "mamba_inject",
        "mamba_svd_flash",
        "mamba_svd_flash_inject",
        "minimal_aurora",
        "minimal_nova",
        "minimal_prism",
        "minimal_svd",
        "minimal_svd_v2",
        "mixattention_moska",
        "mla",
        "moba",
        "moe_deltakv_paged",
        "nasc",
        "nasc_attention",
        "nasc_injection",
        "nexus",
        "nha",
        "nha_compressor",
        "nsa",
        "nystrom",
        "oha",
        "orion",
        "orion_module",
        "performer",
        "phantom",
        "phantom_module",
        "primal",
        "progressive_token_pruning",
        "projection_factorized_attn",
        "qcqa",
        "rala",
        "reap_moe",
        "regla",
        "routing",
        "rpc",
        "salt_attention",
        "samba_m3",
        "samba_moe",
        "simple_ssm",
        "soa_attn",
        "soa_fused_attn",
        "soa_fused_module",
        "soa_module",
        "span_expanded",
        "spectra_v2",
        "spectral_mamba",
        "spectron",
        "squeezing_heads_distiller",
        "svd",
        "svd_compressed",
        "svd_compressed_module",
        "svd_linear",
        "svd_linear_attn",
        "svd_linear_decay",
        "svd_llm_v2",
        "svd_proj_attn",
        "token_compressed",
        "token_compressed_module",
        "tpa",
        "upi_stuffed_mamba",
        "vig",
        "vspec",
        "wgqa",
        "xattention",
        "your_attention_name",
        "zeros"
    ]

    TASKS = ["sst2"]
    total = len(ATTN_MODELS) * len(TASKS)

    task_queue = mp.Queue()
    idx = 0
    for task in TASKS:
        for attn in ATTN_MODELS:
            idx += 1
            task_queue.put((attn, task, idx))

    log_master(f"Trajectory Matrix: {len(ATTN_MODELS)} Models x {len(TASKS)} Tasks = {total} Runs")
    log_master("Targeting 90-100% GPU Compute Utilization via 3 Isolated OS Processes.")

    NUM_WORKERS = 2
    processes = []
    git_lock = mp.Lock()
    for w_id in range(1, NUM_WORKERS + 1):
        p = mp.Process(target=worker_process_main, args=(w_id, task_queue, total, git_lock), daemon=False)
        p.start()
        processes.append(p)
        time.sleep(3)  # Stagger GPU context initialization

    for p in processes:
        p.join()

    log_master("=" * 60)
    log_master("  ALL MULTI-PROCESS RUNS COMPLETED!")
    log_master("=" * 60)


if __name__ == "__main__":
    main()
