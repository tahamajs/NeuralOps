Module neuralops.runners.run_autonomous_48h_pipeline
====================================================
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

Functions
---------

`auto_commit_and_push(msg: str)`
:   

`build_all_streams()`
:   Builds 6 concurrent task streams covering the entire repository.

`get_vram_and_util()`
:   

`log(msg: str)`
:   

`main()`
:   

`run_single_task(task)`
:   

`run_stream_worker(stream_name, task_list)`
:   Legacy stream worker kept for modular compatibility.

`run_task_worker(task, total_pending, task_idx)`
:   Executes a single task in the shared worker pool, updating global accounting.

`vram_governor_loop()`
:   Monitors GPU VRAM every 30 seconds and dynamically adapts worker pool:
      - RTX 4070 (8GB VRAM): Minimum 4,100 MB (50%), Maximum 7,370 MB (90%)
      - If VRAM drops below 50% -> increase workers by 1 (max 6)
      - If VRAM exceeds 90% -> decrease workers by 1 (min 2), sleep 30s to settle