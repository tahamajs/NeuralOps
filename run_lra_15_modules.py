import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LRA15Suite")

MODULES = [
    "asvd_attention",
    "svd_llm_v2_attention",
    "adaptive_svd_attention",
    "squeezing_heads",
    "gqa_attention",
    "primal_attention",
    "svd_linear_attention",
    "res_svd",
    "lora_attention",
    "squeezing_heads_distill",
    "fusion_attn",
    "fused_svd_kernel",
    "flash_attention3",
    "spectron",
    "mamba_svd_flash"
]

TASKS = ["listops", "imdb"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500, help="Cap steps per run for fast convergence/eval")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    results = []
    csv_path = Path(args.output_dir) / "lra_suite_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=================================================================")
    logger.info("🌟 LRA 15-MODULE BENCHMARK SUITE")
    logger.info(f"Tasks: {TASKS} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len} | Epochs: {args.epochs}")
    logger.info("=================================================================")

    python_bin = sys.executable

    for task in TASKS:
        for module in MODULES:
            logger.info(f"\n>>> Running LRA: Task={task.upper()} | Module={module} <<<")
            mod_out_dir = Path(args.output_dir) / f"lra_{task}_{module}"
            mod_out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                python_bin, "-u", "Codes/framework/train_lra.py",
                "--task", task,
                "--module", module,
                "--output_dir", str(mod_out_dir),
                "--batch_size", str(args.batch_size),
                "--seq_len", str(args.seq_len),
                "--epochs", str(args.epochs),
                "--gradient_checkpointing"
            ]
            if args.max_steps:
                cmd.extend(["--max_steps", str(args.max_steps)])

            t0 = time.perf_counter()
            try:
                env = os.environ.copy()
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                env["PYTHONWARNINGS"] = "ignore"

                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env
                )
                duration = time.perf_counter() - t0

                if res.returncode == 0:
                    # Parse result JSON
                    json_file = mod_out_dir / "lra_result.json"
                    if json_file.exists():
                        with open(json_file) as f:
                            data = json.load(f)
                        acc = data.get("best_accuracy", 0.0)
                        logger.info(f"[{task}/{module}] SUCCESS | Best Acc: {acc:.2f}% (in {duration:.1f}s)")
                        status = "PASS"
                    else:
                        acc = 0.0
                        status = "PASS (No output JSON)"
                    error_msg = ""
                else:
                    stderr = res.stderr[-500:] if res.stderr else "Unknown error"
                    logger.warning(f"[{task}/{module}] FAILED (code {res.returncode}): {stderr}")
                    acc = 0.0
                    status = "FAIL (OOM/Error)" if "out of memory" in stderr.lower() else "FAIL"
                    error_msg = stderr

            except subprocess.TimeoutExpired:
                duration = time.perf_counter() - t0
                logger.warning(f"[{task}/{module}] TIMEOUT after {duration:.1f}s")
                status = "TIMEOUT"
                acc = 0.0
                error_msg = "Timeout 600s exceeded"
            except Exception as e:
                duration = time.perf_counter() - t0
                logger.error(f"[{task}/{module}] EXCEPTION: {e}")
                status = "FAIL"
                acc = 0.0
                error_msg = str(e)

            record = {
                "task": task,
                "module": module,
                "status": status,
                "best_accuracy": acc,
                "duration_sec": round(duration, 1),
                "error": error_msg[:200] if error_msg else ""
            }
            results.append(record)
            pd.DataFrame(results).to_csv(csv_path, index=False)

    logger.info(f"\nFinished LRA Suite! Results written to: {csv_path}")

if __name__ == "__main__":
    main()
