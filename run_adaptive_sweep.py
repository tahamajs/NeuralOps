import subprocess
import sys
import time
from pathlib import Path

adaptive_modules = [
    "adaptive_span_attention",
    "block_transformer_hiba",
    "progressive_token_pruning",
    "eat",
    "lola_attention",
    "span_expanded_attention",
    "squeezing_heads_distill",
    "token_compressed_attention",
    "sink_attention",
]

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
master_log = log_dir / "adaptive_sweep_master.log"

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    print(line, end="", flush=True)
    with open(master_log, "a", encoding="utf-8") as f:
        f.write(line)

log("Starting Full Sequential KD Evaluation for All 9 Adaptive Attention Modules...")

for idx, mod in enumerate(adaptive_modules, 1):
    log("=" * 70)
    log(f"QUEUE [{idx}/{len(adaptive_modules)}] Starting: {mod}")
    log("=" * 70)

    cmd = [
        sys.executable,
        "neuralops/benchmarks/_legacy/kd_quick_test.py",
        "--modules", mod,
        "--model_name", "gpt2",
        "--rank", "32",
        "--methods", "C",
        "--use_cache",
        "--batch_size", "8",
        "--grad_accum_steps", "1",
        "--recovery_steps", "100",
        "--self_train_steps", "100",
        "--learning_rate", "3e-4",
        "--eval_every", "50",
        "--device", "cuda",
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=False)
        dur = time.time() - t0
        log(f"COMPLETED {mod} in {dur:.1f}s (Exit Code: {proc.returncode})")
    except Exception as e:
        log(f"ERROR executing {mod}: {e}")

log("=" * 70)
log("ALL ADAPTIVE ATTENTION MODULES COMPLETED EVALUATION!")
log("=" * 70)
