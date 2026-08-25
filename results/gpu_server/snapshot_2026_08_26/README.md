# Partial GPU Sweep Snapshot (2026-08-26)

> **DISCLAIMER:** This is a partial snapshot of an ongoing sweep, not the final benchmark result.

### Snapshot Metadata
* **Execution Date:** 2026-08-26
* **Hardware Device:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM, Compute Capability 8.9)
* **Execution Environment:** Windows OpenSSH Server / PowerShell / CUDA 12.7 / PyTorch 2.6.0+cu124
* **Hyperparameters:**
  * `batch_size`: 128
  * `seq_len`: 512
  * `precision`: BF16 (bfloat16 AMP)
  * `max_steps`: 5 timed passes (+ 2 warmup)
  * `timeout`: 120s hard ceiling per module
* **Sweep Progress:** 61 / 720 combinations completed (60 PASS, 1 TIMEOUT, 0 FAIL)

### Included Files
* `summary.md`: Clean, parsed execution table and status breakdown for all completed evaluations.
* `benchmark_summary.csv`: Machine-readable CSV metrics recorded so far by `results_saver`.
