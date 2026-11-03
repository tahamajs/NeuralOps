---
name: neuralops-benchmark-matrix-orchestrator
description: Multi-phase benchmark orchestration across shapes, dtypes, causal masking, batch saturation, and heterogeneous GPU hardware targets.
---

# NeuralOps Benchmark Matrix Orchestrator Protocol

Single-point microbenchmarks (e.g., $B=1, T=1024$) often mask severe latency cliffs, register thrashing, or memory bandwidth saturation limits that manifest only at scale. Comprehensive performance validation requires orchestrating a multi-dimensional benchmarking matrix across hardware targets.

This skill provides an exhaustive 5-phase protocol for architecting, executing, and analyzing benchmark grids.

---

## Phase 1: Benchmark Matrix Dimension Design
1. **Define Multi-Axis Parameter Grid**:
   - **Sequence Lengths ($T$)**: $\{128, 512, 1024, 2048, 4096, 8192, 16384, 32768\}$
   - **Batch Sizes ($B$)**: $\{1, 2, 4, 8, 16, 32, 64\}$
   - **Hidden Dimensions ($D$)**: $\{64, 128, 256, 512, 1024, 2048\}$
   - **Data Types (`dtype`)**: `torch.float16`, `torch.bfloat16`, `torch.float32`
   - **Masking Modes**: Causal vs Non-Causal (Bidirectional)
2. **Workload Categorization**:
   - Categorize test points into Prefill regime ($M \ge 512$) vs Decode regime ($M = 1, T \ge 1024$).

---

## Phase 2: Timing Methodology & Cold-Cache Isolation
1. **GPU Synchronous Event Timing**:
   - Utilize CUDA Events for precise device-side timestamping:
     ```python
     start_event = torch.cuda.Event(enable_timing=True)
     end_event = torch.cuda.Event(enable_timing=True)
     ```
2. **L2 Cache Eviction Protocol**:
   - Allocate a dummy memory buffer $C_{\text{flush}}$ exceeding the hardware L2 cache capacity (e.g., $256\text{ MB}$ for Ampere/Ada):
     ```python
     cache_flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device='cuda')
     # Prior to timed execution:
     cache_flush.zero_()
     ```
3. **Statistical Aggregation**:
   - Warmup: 10 non-timed iterations.
   - Timed iterations: 50 runs.
   - Report Median, 10th percentile, 90th percentile, and standard deviation.

---

## Phase 3: Hardware Target Execution (Sharif HPC PBS)
1. **PBS Job Parameterization**:
   - Generate batch scripts targeting specific queues (`N3090` for RTX 3090, `titan` for Titan RTX):
     ```bash
     #PBS -N bench_matrix
     #PBS -q N3090
     #PBS -l nodes=1:ppn=8
     #PBS -l walltime=04:00:00
     ```
2. **Environment Verification**:
   - Check environment prior to launching benchmark scripts:
     ```bash
     source /share/apps/Anaconda/anaconda3.8/bin/activate neuralops_cu113
     nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv
     ```

---

## Phase 4: Metric Extraction & Ledger Formatting
1. **Compute Derived Performance Metrics**:
   - **Throughput**: $\text{Tokens/sec} = \frac{B \times T}{\text{Latency (sec)}}$
   - **Arithmetic Intensity**: $\text{TFLOPs} = \frac{\text{Theoretical FLOPs}}{\text{Latency (sec)} \times 10^{12}}$
   - **Memory Bandwidth**: $\text{Effective BW (GB/s)} = \frac{\text{Bytes Transferred}}{\text{Latency (sec)} \times 10^9}$
   - **Speedup Ratio**: $\text{Speedup} = \frac{\text{Latency}_{\text{PyTorch}}}{\text{Latency}_{\text{Kernel}}}$
2. **Structured Ledger Output**:
   - Ingest results directly into SQLite or CSV records with full hardware metadata.

---

## Phase 5: Gating & Anomaly Detection
1. **Performance Regression Gate**:
   - Flag any kernel where $\text{Speedup} < 1.0\times$ across prefill regimes ($T \ge 1024$).
   - Identify whether slowdown is attributable to launch latency, tile size, or driver PTX fallback.
2. **Report Generation**:
   - Generate summary comparison tables highlighting optimal operating regimes for each kernel.
