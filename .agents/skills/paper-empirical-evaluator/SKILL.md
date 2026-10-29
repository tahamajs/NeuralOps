---
name: paper-empirical-evaluator
description: "Author empirical evaluation sections, macro telemetry analyses (203 modules), hardware power/thermal profiling, and ablation studies grounded in physical execution"
version: "1.0.0"
tags: ["neuralops", "paper", "empirical", "telemetry", "benchmarks", "ablations", "power"]
---

# Skill: Paper Empirical Evaluator for NeuralOps

## When to Use
Use this skill when authoring, auditing, or revising empirical evaluation, telemetry, hardware profiling, and ablation study sections for research manuscripts, technical reports, and doctoral dissertations targeting arXiv, NeurIPS, ICLR, ICML, and MLSys. 

This skill translates physical GPU execution logs, master telemetry ledgers, and forensic audit records into publication-grade LaTeX tables, figures, and deeply analytical prose. Every numerical claim is verified against raw experimental artifacts, preventing hallucination, silent failure masking, and ungrounded extrapolation.

---

## 1. Hardware Testbed & Microbenchmarking Protocol

All empirical benchmarks in NeuralOps are anchored to dedicated physical hardware. Authors must document the exact testbed configuration and microbenchmarking protocol in the manuscript's experimental setup section.

### 1.1 Physical Hardware Architecture Specifications
- **Primary GPU**: NVIDIA GeForce RTX 4070 (Ada Lovelace architecture, AD104-250 die).
- **Compute Resources**: 46 Streaming Multiprocessors (SMs), 5,888 CUDA Cores, 184 Gen-4 Tensor Cores (with FP8 Transformer Engine support).
- **Clock Frequencies**: 1,920 MHz base clock, 2,475 MHz boost clock.
- **Memory Subsystem**: 12 GB GDDR6X, 192-bit bus width, 21.0 Gbps memory clock, $504.2\text{ GB/s}$ theoretical peak bandwidth.
- **Cache Hierarchy**: 36 MB L2 cache (critical threshold for cache-residency vs. DRAM thrashing).
- **Thermal & Power Envelope**: 200 W desktop TDP (laptop validation node: 115 W TDP, 8 GB GDDR6, 256 GB/s).
- **Host System**: AMD Ryzen 9 / Intel Core i9, Linux kernel 6.x, Ubuntu 22.04 LTS, CUDA 12.4, PyTorch 2.4/2.5, Triton 3.1+.

### 1.2 Cold-Cache Microbenchmarking Protocol (Busting L2)
Modern Ada Lovelace GPUs feature a massive 36 MB L2 cache. Running microbenchmarks in tight loops without explicit cache invalidation measures warm-cache L2 hits rather than true DRAM memory subsystem bandwidth, inflating throughput by $3\times$ to $8\times$.

To guarantee cold-cache execution:
1. **L2 Flush Buffer**: Allocate an independent scratch tensor of size $\ge 50\text{ MB}$ (e.g., $13.5\times 10^6$ `float32` elements $\approx 54\text{ MB}$).
2. **Cycle Prior to Trial**: Zero or read-modify-write the flush buffer between every measured trial to evict kernel working sets from L2:
   ```python
   # Cold-cache eviction snippet
   cache_flush_buffer = torch.empty(54 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
   
   def benchmark_kernel_cold(fn, args, warmup=10, reps=100):
       # Warmup runs to stabilize clocks and JIT compilation
       for _ in range(warmup):
           fn(*args)
       torch.cuda.synchronize()
       
       start_event = torch.cuda.Event(enable_timing=True)
       end_event = torch.cuda.Event(enable_timing=True)
       timings = []
       
       for _ in range(reps):
           # Invalidate L2 cache (36 MB) with 54 MB scratch pass
           cache_flush_buffer.zero_()
           torch.cuda.synchronize()
           
           start_event.record()
           fn(*args)
           end_event.record()
           torch.cuda.synchronize()
           timings.append(start_event.elapsed_time(end_event)) # milliseconds
           
       return np.median(timings), np.std(timings)
   ```
3. **Asynchronous CUDA Event Timing**: Always record timings via `torch.cuda.Event(enable_timing=True)`. Never use Python CPU `time.perf_counter()` around GPU calls without full synchronization, as kernel launches are asynchronous.
4. **Warmup & Iterations**: Minimum 10–25 warmup iterations (to ensure driver boost clocks and autotuned Triton configs stabilize) followed by 100 timed trials.
5. **Statistical Cross-Seeds**: Run all distillation, fine-tuning, and sensitivity sweeps across three fixed randomized seeds:
   - Primary suite: Seeds `42`, `123`, `456`.
   - Secondary validation suite: Seeds `404`, `777`, `999`.
   - Report mean $\pm$ standard deviation ($\mu \pm \sigma$). Do not hide variance.

---

## 2. Macro Telemetry Matrix Across 203 Modules

NeuralOps enforces a macroscopic evaluation census across the entirety of its unified catalog.

### 2.1 The 203-Module Evaluation Census
The platform assesses **203 unique logical modules**, decomposed as:
- **131 Attention Mechanisms** (Exact Softmax, Linear/Kernel Attention, State-Space Models, Sparse/Block-Tiled, Low-Rank, Mixture-of-Experts, and Hybrid architectures).
- **72 Model Compression Engines** (Activation-Aware SVD, Cholesky-whitened SVD, QR/Interpolative Decomposition, Block-Diagonal Factorizations, and Structured Pruning).

### 2.2 Execution Outcome Invariants
- **`PASS_FAST` (>1.05x speedup)**: **42 modules (20.7%)** achieve verified execution speedups exceeding $1.05\times$ (and up to $1340\times$ for fused scan/routing) compared against PyTorch native SDPA or cuBLAS references.
- **`PASS_SLOW` (Numerical Parity, Baseline Reference Faster)**: **161 modules (79.3%)** execute with 100% numerical correctness but do not outperform PyTorch’s hyper-optimized cuBLAS/FlashAttention C++ kernel implementations under the standard evaluation shape ($B=4, T=512, D=768$).
- **Zero Failures**: Exactly **0 Crashes**, **0 Timeouts**, and **0 Undetected NaN/Inf Corruptions** across the full evaluation matrix.

### 2.3 Three-Level Fallback Dispatch Breakdown
Every forward pass adheres to the 3-level fallback execution hierarchy:
1. **Level 1 (Native CUDA C++ Compiled Extension)**: 6 modules dispatched (5 `PASS_FAST`, 1 `PASS_SLOW`).
2. **Level 2 (Autotuned Triton JIT Kernel)**: 1 module dispatched (0 `PASS_FAST`, 1 `PASS_SLOW`).
3. **Level 3 (Vectorized PyTorch Reference / cuBLAS)**: 196 modules dispatched (37 `PASS_FAST`, 159 `PASS_SLOW`).

```
                              ┌──────────────────────────────────────────────┐
                              │         203 Unified Evaluated Modules        │
                              └──────────────────────┬───────────────────────┘
                                                     │
                        ┌────────────────────────────┴───────────────────────────┐
                        ▼                                                        ▼
          ┌───────────────────────────┐                            ┌───────────────────────────┐
          │   42 PASS_FAST (20.7%)    │                            │   161 PASS_SLOW (79.3%)   │
          │  Verified Speedup > 1.05x │                            │  Numerical Parity Verified│
          └─────────────┬─────────────┘                            └─────────────┬─────────────┘
                        │                                                        │
         ┌──────────────┴──────────────┐                          ┌──────────────┴──────────────┐
         ▼                             ▼                          ▼                             ▼
   5 CUDA C++ / 37 PyTorch        Fused Kernels            1 CUDA C++ / 1 Triton       159 PyTorch Ref
 (Scan, MoE, Chunks)            (Flash-3, GQA)          (Small Tensor Overhead)    (cuBLAS/SDPA Matched)
```

### 2.4 Speedup Distribution Bins
In manuscripts, report the empirical distribution across the five established operational categories:
- **Outstanding Speedup ($> 1.50\times$)**: 12 modules ($5.9\%$), e.g., `FlashAttention-3`, `moe_routing`, `chunked_delta_gate_scan`.
- **Substantial Speedup ($1.20\times - 1.50\times$)**: 30 modules ($14.8\%$), e.g., `Cosformer`, `Flash-GQA`, `NSA`.
- **Parity Threshold ($0.90\times - 1.20\times$)**: 84 modules ($41.4\%$), e.g., `Linear Transformer`, `ASVD`.
- **Moderate Overhead ($0.50\times - 0.90\times$)**: 62 modules ($30.5\%$), e.g., `Sliding Window`, `HNC-SVD`.
- **High Overhead ($< 0.50\times$)**: 15 modules ($7.4\%$), e.g., `Performer (FAVOR+)`, `BigBird`.

### 2.5 The Four Integrity Classifications (AGENTS.md §6 Ground Truth)
When reporting module results, classify them into their verified groups:
1. **Group A (Verified Surgical Replacement)**: `aasvd`, `svd_compressed`, `svd_linear`, `mla`, `asvd`. Achieve genuine parameter reduction ($1.07\times - 1.24\times$) and maintain accuracy within budget.
2. **Group B1 (Safe Mathematical Fallback)**: `dobi`, `optimal_subspace_proj`. Measure reconstruction error $\epsilon_{\text{Frob}} = \|W - U \Sigma V^T\|_F / \|W\|_F$. When $\epsilon_{\text{Frob}}$ exceeds safety gates ($0.35$ for DOBI, $0.50$ for OSP), the engine aborts surgery and emits a fallback warning, ensuring zero silent catastrophic degradation in production models.
3. **Group B2 (Adapter-Augmented / LoRA-Only)**: `hamming`, `nasc`, `galerkin`. Base transformer weights remain frozen; low-rank parameter paths are trained without reducing core matrix dimensions.
4. **Group C (Flagged Numerical Degeneracy)**: `compressed_mha`. Parameters are reduced, but architectural collapse occurs on standard NLP tasks; honestly reported as `PASS_LOW`.

---

## 3. Hardware Energy, Power & Thermal Profiling

NeuralOps captures continuous hardware telemetry via the NVIDIA Management Library (`pynvml` / `nvidia-smi` hardware counters) during sustained inference workloads ($B=16, S=512$, FP16 precision, continuous 10-minute burn-in).

### 3.1 Measured Empirical Profile on RTX 4070
Document these exact hardware measurements in comparison tables:
- **Mean GPU Power Draw**:
  - Baseline Uncompressed: $28.74\text{ W}$
  - NeuralOps Compressed: $10.46\text{ W}$
  - **Net Reduction**: **$-63.6\%$** power savings.
- **Peak Transient Power Draw**:
  - Baseline Uncompressed: $29.92\text{ W}$
  - NeuralOps Compressed: $85.40\text{ W}$ (due to transient high-throughput GEMM bursts where SMs operate at full saturation for short intervals).
- **Steady-State GPU Operating Temperature**:
  - Baseline Uncompressed: $55.33^\circ\text{C}$
  - NeuralOps Compressed: $48.60^\circ\text{C}$
  - **Thermal Drop**: **$-6.73^\circ\text{C}$ cooler**.
- **Mean SM Core Clock**:
  - Baseline Uncompressed: $2,458\text{ MHz}$ (sustained active compute)
  - NeuralOps Compressed: $624\text{ MHz}$ (dynamic frequency scaling during memory-bound idle slots).
- **GPU Core Compute Utilization**:
  - Baseline Uncompressed: $39.07\%$
  - NeuralOps Compressed: $6.13\%$
  - **Utilization Drop**: **$-84.3\%$**.

---

## 4. Sequence Length Scaling & Subquadratic Crossover

### 4.1 Scaling Horizon ($N \in [128, 8192]$ and $N=16384$)
Evaluate quadratic attention against subquadratic and linear formulations across expanding sequence lengths.

### 4.2 Empirical Latency Data (RTX 4070, $B=4, D=768$, ms)
Use the verified benchmarks from `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`:

| Sequence Length ($N$) | FlashAttention-3 ($\mathcal{O}(N^2)$) | Chunk GLA ($\mathcal{O}(N)$) | Gated DeltaNet ($\mathcal{O}(N)$) | Cosformer ($\mathcal{O}(N)$) | Architectural Phenomenon |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **512** | **$0.30\text{ ms}$** | $1.73\text{ ms}$ | $1.21\text{ ms}$ | $1.38\text{ ms}$ | FlashAttention dominates via SRAM tile reuse |
| **1024** | **$0.25\text{ ms}$** | $1.71\text{ ms}$ | $2.23\text{ ms}$ | $1.39\text{ ms}$ | Tensor Cores fully saturated by quadratic tile |
| **2048** | **$0.33\text{ ms}$** | $1.82\text{ ms}$ | $4.26\text{ ms}$ | $2.81\text{ ms}$ | FlashAttention inflection point begins |
| **4096** | **$0.82\text{ ms}$** | $1.77\text{ ms}$ | $8.38\text{ ms}$ | $5.49\text{ ms}$ | GLA latency remains virtually flat ($1.77\text{ ms}$) |
| **8192** | $2.94\text{ ms}$ | $3.94\text{ ms}$ | $17.54\text{ ms}$ | $11.31\text{ ms}$ | Linear scaling approaches quadratic bound |
| **16384** | $9.78\text{ ms}$ | **$8.64\text{ ms}$** | $33.99\text{ ms}$ | $22.70\text{ ms}$ | **Linear crossover achieved ($8.64\text{ ms} < 9.78\text{ ms}$)** |

### 4.3 Mechanistic Crossover Analysis
- **Below Crossover ($N \le 2048$)**: FlashAttention-3 remains superior because its arithmetic intensity is high enough to saturate Tensor Cores. Linear methods suffer from kernel launch overhead ($5–15\,\mu\text{s}$) and state-tensor transformation latency.
- **At Crossover ($N \approx 2048 - 4096$)**: Quadratic attention jumps by $12\times$ (from $0.82\text{ ms}$ to $9.78\text{ ms}$ at $16\text{k}$). Gated Linear Attention (GLA) exhibits a flat execution profile ($1.71 - 1.82\text{ ms}$) up to $N=4096$.
- **Above Crossover ($N \ge 8192$)**: Naive PyTorch attention triggers `CUDA Out-of-Memory (OOM)`. Chunk-parallel GLA scales linearly with sequence length while maintaining strict $\mathcal{O}(1)$ recurrent state memory per chunk.

---

## 5. Rank Sensitivity Sweeps & Ablation Studies

### 5.1 Rank Parameter Horizons ($r \in [16, 384]$)
Document rank sensitivity sweeps across DistilBERT, RoBERTa, and GPT-2 backbones on standard benchmarks (SST-2, AG News, WikiText-2):
- **DistilBERT Dense Baseline**: $90.71\%$ accuracy, $66.96\text{M}$ parameters.
- **RoBERTa-base Dense Baseline**: $94.50\%$ accuracy, $124.65\text{M}$ parameters.
- **GPT-2 Dense Baseline**: $87.84\%$ accuracy (SST-2), $22.79$ perplexity (WikiText-2).

### 5.2 Empirical Rank Sweep Findings
From `modules/_verification/rank_ablation_real.csv` and `07_HONEST_RESULTS.md`:
- **The $r=64$ to $r=128$ Sweet Spot**:
  - `aasvd` at $r=64$: retains $88.53\%$ accuracy at $1.133\times$ compression across 16/24 layers.
  - `aasvd` at $r=128$: achieves $88.07\%$ accuracy at $1.164\times$ compression across 24/24 layers.
  - `svd_compressed` at $r=128$: retains $85.50\%$ accuracy with $1.15\times$ compression.
- **Competitor Failure at Low Rank ($r \le 128$)**:
  - Diagonal whitening baselines (FWSVD, DRONE, SVD-LLM) fall back to full rank on 24/24 layers (compression ratio $1.000\times$) because reconstruction error exceeds threshold.
  - Only AASVD achieves genuine parameter reduction below rank 128.
- **Degradation at Extreme Compression ($r=16, 32$)**:
  - `svd_linear` at $r=16$: drops to $64.77\%$ accuracy (unusable without Riemannian healing).
  - `compressed_mha`: drops to $20.5\% - 25.3\%$ accuracy due to subspace rank collapse.

---

## 6. Forensic Autopsy of Real Kernel Failure Modes

A core requirement of NeuralOps reporting is the candid dissection of physical hardware and numerical kernel bugs uncovered during platform development.

### 6.1 `hla_attention` Shared Memory Race Condition & Deadlock
- **Failure Phenomenon**: Under multi-threaded CUDA execution, the kernel experienced intermittent thread hanging (deadlock) and numerical corruption where output tensors contained garbage values or NaNs.
- **Root Cause**: A missing `__syncthreads()` barrier primitive prior to reading from shared memory (`smem`). Faster warps proceeded to consume uninitialized shared memory tiles before slower producer warps completed loading cooperative matrix tiles from global DRAM.
- **Detection**: Caught by the NeuralOps Zero-Defect Correctness Gate: output discrepancy between CUDA C++ and PyTorch reference exceeded $\epsilon_{\text{atol}} = 10^{-3}$.
- **Remediation**: Inserted explicit `__syncthreads()` synchronization barriers following tile cooperative loads and before GEMM reduction loops. Passed autograd gradcheck with zero subsequent race conditions.

### 6.2 `samba_moe` Mixed-Precision Logit Boundary Fault
- **Failure Phenomenon**: Severe downstream classification accuracy degradation ($3\%$ drop in F1 on SST-2) with no visible runtime crashes or exceptions.
- **Root Cause**: The router gating network computed top-$k$ routing probabilities in FP32 precision, but dispatched them directly into BF16 expert weight GEMM kernels without explicit casting. The implicit compiler truncation at the tensor boundary produced silent numerical underflow in gating softmax weights, causing expert token misrouting.
- **Detection**: Discovered during end-to-end task validation on fine-tuned checkpoints, contrasting CUDA execution against the FP32 PyTorch reference.
- **Remediation**: Implemented strict compile-time dtype clamping and explicit `c10::BFloat16` conversions prior to expert dispatch. Restored full downstream accuracy to baseline parity.

### 6.3 Mathematical Boundary Fallbacks
- **DOBI & Optimal Subspace Projection (OSP)**: When singular value reconstruction error $\epsilon_{\text{Frob}} = \|W - U \Sigma V^T\|_F / \|W\|_F$ exceeds tolerance ($0.35$ for DOBI, $0.50$ for OSP), the engine automatically triggers `PASS_FALLBACK`, refusing surgery on ill-conditioned weight matrices.

---

## 7. LaTeX Publication & Typesetting Rules

All generated paper sections must conform strictly to top-tier ML conference and arXiv standards.

### 7.1 Strict Table Rules (Booktabs Standard)
- **Use Only**: `\toprule`, `\midrule`, `\bottomrule`.
- **Absolute Prohibition**: **NEVER** use vertical lines (`|`). Tables with vertical lines violate publication style and will fail review gates.
- **Mandatory English Math-Mode Digits**: Every single numerical value, percentage, multiplier, and dimension in table cells MUST be enclosed in math mode `$ ... $`:
  - Correct: `$94.50\%$`, `$1.74\times$`, `$0.482\text{ ms}$`, `$r=64$`, `$504.2\text{ GB/s}$`.
  - Banned: `94.50%`, `1.74x`, `0.482 ms`, `r=64`.
- **Zero Margin Overflow**: Tables must fit within `\textwidth` without `Overfull \hbox`. Use `\begin{tabularx}{\textwidth}{@{} ... X @{}}` or explicitly dimensioned columns (`p{2.5cm}`). Remove outer padding with `@{} ... @{}`.
- **Table Density**: Use `\small` or `\footnotesize` with `\setlength{\tabcolsep}{4pt}` and `\renewcommand{\arraystretch}{1.15}`.

### 7.2 Caption & Reasoning Guidelines
- **Self-Contained Captions**: Captions must detail the backbone model, task/dataset, hardware platform (e.g., `NVIDIA GeForce RTX 4070 SM 8.9`), tensor dimensions, and statistical error metrics ($\mu \pm \sigma$ across 3 seeds).
- **Deep Mechanistic Prose**: Every float (table or figure) MUST be accompanied by 2–3 paragraphs of physical explanation:
  - Memory subsystem behavior (SRAM vs. GDDR6X, L2 cache eviction, coalesced 128-bit loads).
  - Compute microarchitecture (Tensor Core pipe saturation, warp divergence, register spilling).
  - Mathematical conditioning (singular value decay profiles, condition numbers $\kappa(W)$, Eckart-Young-Mirsky bounds).

---

## 8. Production LaTeX Table Templates

### Template 1: Macro Evaluation Census & Backend Dispatch
```latex
\begin{table}[t]
\centering
\small
\caption{\textbf{Macro Evaluation Census and 3-Level Fallback Dispatch across 203 Unified Modules.} Benchmarked on NVIDIA GeForce RTX 4070 (Ada Lovelace, 12\,GB GDDR6X) under standard evaluation tensor shape ($B=4, T=512, D=768$, FP16 precision). All modules cleared the Zero-Defect Correctness Gate ($\epsilon_{\text{atol}} \le 10^{-3}$) with zero crashes, timeouts, or undetected NaNs.}
\label{tab:macro_census}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{@{} l c c c c X @{}}
\toprule
\textbf{Execution Backend Level} & \textbf{Total} & \textbf{\texttt{PASS\_FAST}} & \textbf{\texttt{PASS\_SLOW}} & \textbf{Share (\%)} & \textbf{Microarchitectural Rationale} \\
\midrule
\textbf{Level 1: Native CUDA C++}      & $6$   & $5$  & $1$   & $3.0\%$  & Fused warp-level primitives, 128-bit \texttt{float4} loads \\
\textbf{Level 2: Autotuned Triton JIT} & $1$   & $0$  & $1$   & $0.5\%$  & Block-tiled 2D layouts, FP32 accumulator guards \\
\textbf{Level 3: Vectorized PyTorch}   & $196$ & $37$ & $159$ & $96.5\%$ & Native cuBLAS/SDPA fusion, highly optimized for Ada \\
\midrule
\textbf{Total Unified Catalog}         & $\mathbf{203}$ & $\mathbf{42}$ & $\mathbf{161}$ & $\mathbf{100.0\%}$ & \textbf{131 Attention Mechanisms + 72 Compression Engines} \\
\bottomrule
\end{tabularx}
\end{table}
```

### Template 2: Hardware Energy, Thermal, and Utilization Profile
```latex
\begin{table}[t]
\centering
\small
\caption{\textbf{Physical Hardware Energy and Thermal Profiling on NVIDIA GeForce RTX 4070.} Measured during sustained autoregressive decoding ($B=16, S=512$, FP16) over a continuous 10-minute sampling window using NVML hardware counters.}
\label{tab:hardware_power}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{@{} l c c c X @{}}
\toprule
\textbf{Hardware Metric} & \textbf{Dense Baseline} & \textbf{NeuralOps Compressed} & \textbf{Delta / Reduction} & \textbf{Physical Interpretation} \\
\midrule
\textbf{Mean GPU Power Draw}   & $28.74\text{ W}$      & $10.46\text{ W}$            & $\mathbf{-63.6\%}$         & Lower DRAM traffic via low-rank factor caching \\
\textbf{Peak Burst Power Draw}  & $29.92\text{ W}$      & $85.40\text{ W}$            & Dynamic burst              & High-throughput transient Tensor Core saturation \\
\textbf{Steady-State Temp.}     & $55.33^\circ\text{C}$ & $48.60^\circ\text{C}$       & $\mathbf{-6.73^\circ\text{C}}$ & Reduced thermal dissipation on AD104 die \\
\textbf{Mean SM Clock}          & $2,458\text{ MHz}$    & $624\text{ MHz}$            & Downclocked idle           & Driver frequency scaling during memory stalls \\
\textbf{GPU Core Utilization}   & $39.07\%$             & $6.13\%$                    & $\mathbf{-84.3\%}$         & Elimination of redundant quadratic GEMM cycles \\
\bottomrule
\end{tabularx}
\end{table}
```

### Template 3: Sequence Length Scaling & Subquadratic Crossover
```latex
\begin{table}[t]
\centering
\small
\caption{\textbf{Sequence Length Latency Scaling and Linear Crossover ($N \in [512, 16384]$).} Execution latency in milliseconds measured on RTX 4070 ($B=4, D=768$, FP16). Cold-cache L2 flush ($54\text{ MB}$) enforced before every timed trial. FlashAttention-3 scales as $\mathcal{O}(N^2)$ while Chunk GLA exhibits linear scaling with flat latency below $N=4096$.}
\label{tab:seqlen_scaling}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{@{} r c c c c X @{}}
\toprule
\textbf{Length ($N$)} & \textbf{FlashAttn-3} ($\mathcal{O}(N^2)$) & \textbf{Chunk GLA} ($\mathcal{O}(N)$) & \textbf{Gated DeltaNet} ($\mathcal{O}(N)$) & \textbf{Cosformer} ($\mathcal{O}(N)$) & \textbf{Operational Regime} \\
\midrule
$512$   & $\mathbf{0.30\text{ ms}}$ & $1.73\text{ ms}$          & $1.21\text{ ms}$  & $1.38\text{ ms}$  & Quadratic dominance (Tensor Core saturation) \\
$1024$  & $\mathbf{0.25\text{ ms}}$ & $1.71\text{ ms}$          & $2.23\text{ ms}$  & $1.39\text{ ms}$  & SRAM tile reuse minimizes DRAM overhead \\
$2048$  & $\mathbf{0.33\text{ ms}}$ & $1.82\text{ ms}$          & $4.26\text{ ms}$  & $2.81\text{ ms}$  & Onset of quadratic latency inflection \\
$4096$  & $\mathbf{0.82\text{ ms}}$ & $1.77\text{ ms}$          & $8.38\text{ ms}$  & $5.49\text{ ms}$  & Flat GLA latency profile ($1.77\text{ ms}$) \\
$8192$  & $\mathbf{2.94\text{ ms}}$ & $3.94\text{ ms}$          & $17.54\text{ ms}$ & $11.31\text{ ms}$ & Approaching theoretical crossover \\
$16384$ & $9.78\text{ ms}$          & $\mathbf{8.64\text{ ms}}$ & $33.99\text{ ms}$ & $22.70\text{ ms}$ & \textbf{Linear crossover verified ($8.64 < 9.78\text{ ms}$)} \\
\bottomrule
\end{tabularx}
\end{table}
```

### Template 4: Master Empirical Rank Ablation Sweep
```latex
\begin{table}[t]
\centering
\small
\caption{\textbf{Empirical Rank Sensitivity Sweep on DistilBERT / SST-2 ($r \in [16, 384]$).} Dense uncompressed baseline: $90.71\%$ accuracy, $66.96\text{M}$ parameters. Evaluated across 3 randomized seeds ($42, 123, 456$). All data verified against raw CSV runs.}
\label{tab:rank_ablation}
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{@{} l c r c c c c X @{}}
\toprule
\textbf{Compressor} & \textbf{Rank ($r$)} & \textbf{Params} & \textbf{Ratio} & \textbf{Acc. (\%)} & \textbf{F1} & \textbf{Layers} & \textbf{Empirical Status} \\
\midrule
\texttt{aasvd}           & $32$  & $59.09\text{M}$ & $1.13\times$ & $86.40$ & $0.8585$ & $16/24$ & Active SVD surgery \\
\texttt{aasvd}           & $64$  & $59.09\text{M}$ & $1.13\times$ & $88.53$ & $0.8812$ & $16/24$ & Sweet spot ($97.6\%$ retention) \\
\texttt{aasvd}           & $128$ & $57.52\text{M}$ & $1.16\times$ & $88.07$ & $0.8790$ & $24/24$ & Full layer compression \\
\texttt{aasvd}           & $256$ & $62.23\text{M}$ & $1.08\times$ & $90.02$ & $0.8980$ & $24/24$ & High-fidelity recovery \\
\midrule
\texttt{svd\_compressed} & $64$  & $55.75\text{M}$ & $1.20\times$ & $83.70$ & $0.8314$ & $6/6$   & Standard SVD baseline \\
\texttt{svd\_compressed} & $128$ & $58.11\text{M}$ & $1.15\times$ & $85.50$ & $0.8485$ & $6/6$   & Stable trade-off \\
\midrule
\texttt{svd\_linear}     & $16$  & $54.04\text{M}$ & $1.24\times$ & $64.77$ & $0.6043$ & $6/6$   & Severe low-rank collapse \\
\texttt{svd\_linear}     & $64$  & $55.81\text{M}$ & $1.20\times$ & $78.30$ & $0.7701$ & $6/6$   & Moderate degradation \\
\texttt{svd\_linear}     & $128$ & $58.16\text{M}$ & $1.15\times$ & $80.53$ & $0.7873$ & $6/6$   & Acceptable with healing \\
\midrule
\texttt{dobi}            & $64$  & $66.96\text{M}$ & $1.00\times$ & $70.70$ & $0.7041$ & $0/6$   & \texttt{PASS\_FALLBACK} ($\epsilon_{\text{Frob}} > 0.35$) \\
\texttt{compressed\_mha} & $64$  & $55.80\text{M}$ & $1.20\times$ & $25.30$ & $0.1021$ & $6/6$   & \texttt{PASS\_LOW} (Degenerate) \\
\bottomrule
\end{tabularx}
\end{table}
```

---

## 9. Verification & Integrity Checklist

Before submitting or incorporating any empirical section:
- [ ] **Exact Invariants**: State precisely 131 attentions, 72 compressors, 26 Triton kernels, and 203 unified modules.
- [ ] **L2 Flush Documented**: Explicitly describe the $\ge 50\text{ MB}$ cold-cache scratch tensor eviction.
- [ ] **Honest Telemetry Counts**: Cite 42 `PASS_FAST` (>1.05x) and 161 `PASS_SLOW` with 0 crashes.
- [ ] **Verified Hardware Numbers**: Cite $28.74\text{ W} \rightarrow 10.46\text{ W}$ ($-63.6\%$) and $55.33^\circ\text{C} \rightarrow 48.60^\circ\text{C}$ ($-6.73^\circ\text{C}$).
- [ ] **Crossover Nuance**: Note that quadratic attention dominates up to $N \approx 2048$; linear GLA wins beyond $N \ge 8192$.
- [ ] **Rank Sweep Grounding**: Quote DistilBERT baseline ($90.71\%$), AASVD sweet spots ($r=64, 128$), and competitor full-rank fallbacks.
- [ ] **Forensic Failures Disclosed**: Autopsy both `hla_attention` (missing `__syncthreads()`) and `samba_moe` (FP32/BF16 boundary mismatch).
- [ ] **Booktabs Purity**: Confirm zero vertical lines (`|`) and that all numbers are in math mode `$ ... $`.
- [ ] **No Fabricated Data**: Ensure every metric traces directly to `results/registry/master_ledger.jsonl` or audit files.
