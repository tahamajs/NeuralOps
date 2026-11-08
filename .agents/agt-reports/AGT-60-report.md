# AGT-60 Execution Report: Finalize and Execute Full 3090 Benchmark Sweep

**Date:** 2026-11-08  
**Scope:** `scripts/benchmarks/run_3090_full.py`, `scripts/benchmarks/make_paper_artifacts.py`, `paper/tables/`, `paper/figures/`.  
**Host Target:** NVIDIA GeForce RTX 3090 (Sharif HPC Cluster Node `GPU-3090-1`).  
**Status:** **DEFERRED / STOP CONDITION TRIGGERED (Hardware Boundary)**  

---

## Executive Summary
Task **AGT-60** mandates executing the full canonical 369-benchmark sweep on dedicated NVIDIA GeForce RTX 3090 hardware (`GPU-3090-1`) using the multi-worker two-pass orchestration protocol, persisting empirical telemetry manifests, and synthesizing camera-ready paper tables and figures exclusively from physical GPU execution telemetry.

In strict compliance with **`AGENTS.md` §0.1** (Verification honesty: *"Never claim a command, test, kernel, compiler, benchmark, or job succeeded unless it was physically executed and its result was observed"*), **§0.2** (Zero fake artifacts and metrics: *"Never write hard-coded throughput, speedup, accuracy, perplexity, memory, or parity values when real data is missing"*), **§4.1** (Local macOS: *"Local macOS cannot verify CUDA C++, Triton JIT, CUDA gradcheck, or GPU memory behavior; never label local CPU/MPS output as GPU verification"*), and the explicit prompt **Stop Condition** (*"If the 3090 is not present on this host — STOP and report which host has it; do not run on CPU and present it as GPU results"*), physical execution on this workstation was halted.

Per Prompt Instruction Line 12 (*"If a task's stop condition triggers, report it exactly and continue with the next task that does not depend on it — do not stop the whole queue"*), this report formally records the hardware boundary, details the host allocation and deployment scripts, and passes execution to **AGT-61**.

---

## 1. Physical Host & Hardware Audit
- **Current Environment:** Local Development Machine (`tahamajss-MacBook-Air.local`).
- **Operating System:** Darwin 27.2.0 (macOS arm64, Apple Silicon).
- **Physical Accelerator:** Apple M-Series Silicon (MPS / CPU only).
- **CUDA Availability:** `torch.cuda.is_available() == False` (`nvidia-smi` command not found).
- **Target Host for Execution:** Sharif HPC Cluster (`hpc.sharif.edu`), Torque PBS GPU node `GPU-3090-1` (equipped with 24GB VRAM NVIDIA GeForce RTX 3090, CUDA 12.4, SM 8.6 compute capability).

---

## 2. Benchmark Sweep Architecture & Readiness
The benchmark infrastructure validated in preceding tasks (**AGT-56**, **AGT-57**, **AGT-58**, **AGT-59**) guarantees complete readiness for cluster execution:
1. **Canonical Inventory (AGT-57):** Full 369-benchmark registration verified across 131 attention mechanisms, 72 compressor modules, 140 CUDA package units, and 26 Triton JIT modules in `BENCHMARK_REGISTRY`.
2. **Subprocess Isolation & Watchdogs (AGT-58):** Dedicated worker isolation via `multiprocessing.get_context("spawn")` prevents rogue kernel faults (e.g. SIGSEGV, CUDA illegal memory access) from aborting the orchestrator sweep.
3. **Dedicated CLI Subcommands (AGT-59):** Fully decoupled `neuralops benchmark` command group and `compare` gate (`FAIL_NUMERICS = 4`).
4. **Benchmark Matrix Orchestrator (AGT-59):** Multi-dimensional matrix executor (`neuralops.benchmarks.matrix:run_benchmark_matrix`) supporting synchronous GPU event timing, cold L2 cache flushing, percentile latency aggregation, and prefill speedup regression gating.

---

## 3. Remote Cluster Execution Protocol (Deferred to Sharif HPC)
To execute the physical 3090 sweep on Sharif HPC:
1. **Sync Source to HPC:**
   ```bash
   rsync -avz --exclude='.git' --exclude='.venv' /Users/tahamajs/dev/Project/ hpc:~/NeuralOps/
   ```
2. **Submit Torque PBS Job:**
   ```bash
   ssh hpc "qsub scripts/hpc/job_3090_full_sweep.pbs"
   ```
3. **Generate Paper Tables and Artifacts:**
   Upon job completion on `GPU-3090-1`, telemetry manifests will be ingested using `scripts/benchmarks/make_paper_artifacts.py` into `paper/tables/` and `paper/figures/`.

---

## 4. Conclusion & Next Step
- Hardware boundary respected with zero fake metrics.
- Proceeding immediately to **AGT-61** (Version 2.1.0 release bump, CHANGELOG [2.1.0], and packaging verification).
