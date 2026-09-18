#!/usr/bin/env python3
"""
Parallel dataset completion runner with checkpointing and auto-resume.
- Evaluates AG News and MNLI runs on DistilBERT, RoBERTa, and GPT-2.
- Uses multiprocessing to run N workers in parallel on the GPU.
- Records completed jobs in an append-only JSONL checkpoint file.
- Automatic skip of completed jobs on restart.
"""
import os, sys, json, time, subprocess, argparse
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime

CKPT_FILE = None
HF_REPO = "svd-linear-attention-results"

# Map model shorthand names to checkpoint directories on disk
MODEL_PATH_MAP = {
    "distilbert": "models/distilbert_sst2",
    "roberta": "models/roberta-base",
    "gpt2": "models/gpt2"
}

def init_worker(ckpt_path):
    global CKPT_FILE
    CKPT_FILE = Path(ckpt_path)

def load_completed():
    if not CKPT_FILE or not CKPT_FILE.exists():
        return set()
    done = set()
    with open(CKPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("status") in ("OK", "PASS", "PASS_NO_COMPRESSION"):
                    done.add((d["module"], d["model"]))
            except Exception:
                pass
    return done

def mark_completed(module, model, status, extra=None):
    rec = {
        "module": module,
        "model": model,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        **(extra or {})
    }
    with open(CKPT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

def run_one(job):
    module, model, dataset, config, ckpt_path = job
    global CKPT_FILE
    if CKPT_FILE is None:
        CKPT_FILE = Path(ckpt_path)

    out_dir = Path(f"results/{dataset}_unified_{module}/{model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs/parallel_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset}_{module}_{model}.log"

    # Map dataset string to training script parameter
    ds_name = "ag_news" if dataset == "agnews" else "mnli"

    # Resolve model directory path
    model_arg = MODEL_PATH_MAP.get(model, model)
    if not os.path.exists(model_arg):
        model_arg = model

    # Execution script call
    cmd = [
        sys.executable, "-u", "Codes/framework/run_quality_sweep.py",
        "--modules", module,
        "--model", model_arg,
        "--dataset", ds_name,
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--finetune_steps", str(config["finetune_steps"]),
        "--seq_len", str(config["seq_len"]),
        "--seed", str(config["seed"]),
        "--num_labels", str(config["num_labels"]),
        "--output_dir", str(out_dir),
    ]

    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True, timeout=1800)

        res_file = out_dir / "finetune_metadata.json"
        metrics_file = list(out_dir.glob("*_full_metrics.json"))
        acc = None
        f1 = None
        vram = None

        if metrics_file and metrics_file[0].exists():
            try:
                m_data = json.load(open(metrics_file[0]))
                acc = m_data.get("performance", {}).get("accuracy_pct")
                f1 = m_data.get("performance", {}).get("f1_score")
                vram = m_data.get("hardware", {}).get("peak_vram_mb")
            except Exception:
                pass

        if res_file.exists():
            mark_completed(module, model, "OK", {
                "accuracy": acc,
                "f1": f1,
                "peak_vram_mb": vram
            })
            return (module, model, "OK")
        else:
            mark_completed(module, model, "NO_OUTPUT")
            return (module, model, "NO_OUTPUT")
    except subprocess.TimeoutExpired:
        mark_completed(module, model, "TIMEOUT")
        return (module, model, "TIMEOUT")
    except subprocess.CalledProcessError as e:
        mark_completed(module, model, "ERROR", {"rc": e.returncode})
        return (module, model, "ERROR")
    except Exception as e:
        mark_completed(module, model, "EXCEPTION", {"err": str(e)})
        return (module, model, "EXCEPTION")

def main():
    global CKPT_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["agnews", "mnli"])
    ap.add_argument("--workers", type=int, default=2, help="parallel workers on single GPU")
    ap.add_argument("--hf_push_every", type=int, default=100)
    args = ap.parse_args()

    CKPT_FILE = Path(f"audit/{args.dataset}_completion_2026_09_18/checkpoint.jsonl")
    CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rem_file = Path(f"audit/{args.dataset}_completion_2026_09_18/remaining.txt")
    modules = [l.strip() for l in rem_file.read_text().splitlines() if l.strip()]
    config = json.load(open(f"audit/{args.dataset}_completion_2026_09_18/config_lock.json"))[args.dataset]

    models = ["distilbert", "roberta", "gpt2"]

    done = load_completed()
    jobs = []
    for m in modules:
        for model in models:
            if (m, model) not in done:
                jobs.append((m, model, args.dataset, config, str(CKPT_FILE)))

    print(f"[START] {args.dataset}: {len(jobs)} jobs remaining, {len(done)} already done, {args.workers} workers", flush=True)

    completed_count = 0
    with Pool(args.workers, initializer=init_worker, initargs=(str(CKPT_FILE),)) as pool:
        for result in pool.imap_unordered(run_one, jobs):
            completed_count += 1
            mod, mdl, status = result
            print(f"[{completed_count}/{len(jobs)}] {mod}/{mdl} -> {status}", flush=True)

            if completed_count % args.hf_push_every == 0:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    api.upload_folder(
                        folder_path=f"results/{args.dataset}_unified_{mod}/",
                        path_in_repo=f"{args.dataset}_unified_{mod}/",
                        repo_id=HF_REPO,
                        repo_type="dataset"
                    )
                except Exception as e:
                    print(f"[HF_PUSH_WARNING] {e}", flush=True)

if __name__ == "__main__":
    main()
