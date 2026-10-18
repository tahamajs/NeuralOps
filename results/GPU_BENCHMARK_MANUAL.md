# 🚀 NeuralOps v1.4.0 GPU Hardware Benchmark Execution & Verification Manual

**Target Platform:** Remote Dedicated GPU Server `taha@192.168.1.100`  
**GPU Architecture:** NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace, SM 8.9, 8,192 MB GDDR6)  
**Host Paths:** `/mnt/d/SVD_Project` and `/mnt/d/SVD_linear_Attention`  
**Python Runtime:** `/mnt/d/venv/bin/python3` (PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.2.x)

---

## 📌 Overview & Operating Laws

To ensure 100% empirical reproducibility and numerical parity across our **ICLR 2027** and **NeurIPS 2026** submissions, all benchmark suites, micro-benchmarks, and scaling experiments must execute strictly on physical GPU hardware.

> [!IMPORTANT]
> **Strict Law**: Never execute benchmark Python scripts locally on macOS. Always invoke them via SSH inside `/mnt/d/venv/bin/python3` on `taha@192.168.1.100`.

---

## 🛠️ Complete Benchmark Execution Roadmap

### 1. Invariant Registry & Baseline Sanity Check
Before launching heavy GPU workloads, verify that the package registry preserves the canonical counts:
```bash
ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && /mnt/d/venv/bin/python3 scripts/utils/verify_readme_counts.py"
```
*Expected Output:*
`README counts verified successfully (v1.4.0): 131 attention mechanisms, 72 compression engines, 26 Triton kernels.`

---

### 2. SOTA Gap Closure & Parity Suite (5 Key Empirical Proofs)
Executes the closed gap suite (SVD-LLM v2 activation truncation, ASVD KV-cache streaming, FP8 e4m3fn bounding, DSA learned block-sparse routing, and REAP distributed MoE pruning):
```bash
ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && PYTHONPATH=. /mnt/d/venv/bin/python3 scripts/benchmarks/benchmark_sota.py"
```
*Outputs Generated:*
- `results/SOTA_GAP_CLOSURE_REPORT.md` (100% PASS verification metrics)

---

### 3. Comprehensive Master Benchmark Suite (All 131 Attentions)
Runs head-to-head evaluation across DistilBERT, RoBERTa, and GPT-2:
```bash
ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && PYTHONPATH=. /mnt/d/venv/bin/python3 scripts/benchmarks/benchmark_all_131_attentions.py"
```
*Outputs Generated:*
- `results/accuracy_and_ppl_benchmark_all131.json`
- `results/triton_all_131_attentions_benchmark.json`

---

### 4. Matrix Compression Engines Benchmark (All 72 Compressors)
Evaluates low-rank SVD, Tucker, Tensor-Train, Kronecker, and structured pruning factorizers:
```bash
ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && PYTHONPATH=. /mnt/d/venv/bin/python3 scripts/benchmarks/benchmark_all_72_compressors.py"
```
*Outputs Generated:*
- `results/triton_all_72_compressors_benchmark.json`
- `results/v13_deep_empirical_suite/roberta_72_compressors/results.json`

---

### 5. 3-Level Latency Hierarchy Benchmark (Native CUDA vs. Triton JIT vs. PyTorch)
Profiles wall-clock execution across the 3-level fallback hierarchy over 100 timed iterations following 25 warmups:
```bash
ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && PYTHONPATH=. /mnt/d/venv/bin/python3 scripts/benchmarks/benchmark_3level_cuda_triton_pytorch.py"
```
*Outputs Generated:*
- `results/benchmark_3level_cuda_triton_pytorch.csv`
- Directly feeds Table 18 in `paper/*/tables/table_18_native_cuda_kernels_3level.tex`

---

### 6. Triton GPU Acceleration Leaderboard (All 26 Flagship Kernels)
Measures block dimensions, shared memory footprints, register allocations, and speedups:
```bash
ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && PYTHONPATH=. /mnt/d/venv/bin/python3 scripts/benchmarks/benchmark_all_26_kernels.py"
```
*Outputs Generated:*
- `results/triton_all_26_kernels_benchmark.json`
- Directly feeds Table 13 in `paper/*/tables/table_13_flagship_triton_kernels.tex`

---

### 7. Sequence Length Context Scaling Suite ($T \in [128, 8192]$)
Evaluates latency and peak VRAM scaling under sequence lengths $T \in \{512, 1024, 2048, 4096, 8192\}$:
```bash
ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && PYTHONPATH=. /mnt/d/venv/bin/python3 scripts/benchmarks/run_fla_head_to_head_benchmark.py"
```
*Outputs Generated:*
- `results/SEQLEN_SCALING_MASTER.csv`
- Directly feeds Table 7 in `paper/*/tables/table7_long_context.tex`

---

### 8. Master Empirical Ingestion & SQLite Ledger Rebuild
Consolidates all freshly benchmarked runs into the canonical master database:
```bash
ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && PYTHONPATH=. /mnt/d/venv/bin/python3 scripts/utils/ingest_historical_results.py"
```
*Outputs Generated:*
- `results/registry/master_ledger.sqlite`
- `results/registry/master_ledger.jsonl`

---

## 📊 Dual Mirroring & Synchronization Commands
After running any benchmark on the remote GPU server, ensure both mirror workspaces are 100% in sync:
```bash
# Sync SVD_Project to local workspace
rsync -avz taha@192.168.1.100:/mnt/d/SVD_Project/results/ results/

# Sync SVD_Project to SVD_linear_Attention
ssh taha@192.168.1.100 "rsync -avz /mnt/d/SVD_Project/results/ /mnt/d/SVD_linear_Attention/results/"
```

---

## 🎯 Verification Criteria for Camera-Ready Submission
- **Exact Counts Check**: Exactly 131 attention mechanisms, 72 compression engines, and 26 Triton kernels.
- **Zero Placeholder Check**: Zero occurrences of `TODO&#58; Identify specific limitations` or `Pending &#124; Pending` in any results file or LaTeX table.
- **Numerical Parity**: Max relative error $< 0.05$ for streaming approximations and $\Delta_{\max} < 0.01$ for full rank factorizations.
