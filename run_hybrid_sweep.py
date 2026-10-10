import subprocess
import sys
import time
from pathlib import Path

modules = [
    "regla_attention",
    "fusion_attn",
    "gated_hybrid_kernel",
    "granite_hybrid",
    "halo_regla",
    "hybrid_attention",
    "hybrid_local_kernel",
    "landmark_local",
    "mixattention_moska",
    "optimized_hybrid",
    "positive_kernel_linear",
    "refined_landmark",
    "spectron_fp8",
    "split_head_hybrid",
    "tri_branch_hybrid",
    "fusion_attn",  # Re-run with past_key_value autoregressive fix
]

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
master_log = log_dir / "hybrid_sweep_master.log"

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    print(line, end="", flush=True)
    with open(master_log, "a", encoding="utf-8") as f:
        f.write(line)

log("Starting Automated Hybrid Sweep across 15 hybrid attention modules...")

for idx, mod in enumerate(modules, 1):
    log(f"======================================================================")
    log(f"PROGRESS: [{idx}/{len(modules)}] Starting Module: {mod}")
    log(f"======================================================================")
    
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
        "--recovery_steps", "300",
        "--self_train_steps", "200",
        "--learning_rate", "3e-4",
        "--eval_every", "100",
        "--device", "cuda",
    ]
    
    start_t = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=False, check=False)
        dur = time.time() - start_t
        log(f"COMPLETED Module: {mod} in {dur:.1f}s (Exit Code: {proc.returncode})")
    except Exception as e:
        log(f"ERROR running Module {mod}: {e}")

log("ALL HYBRID MODULES FINISHED!")
