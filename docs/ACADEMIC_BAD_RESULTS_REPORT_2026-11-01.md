# Academic & Empirical Bad Results Audit Report

**Date:** 2026-11-01
**Auditor / Author:** `taha <tahamaj4@gmail.com>`
**Scope:** Forensic inventory and cross-referencing of negative results, failure modes, timeouts, sub-1.0x kernel latencies, unverified states, and pseudo-Triton functions across `academic/`, `docs/`, `results/`, `README.md`, and `CHANGELOG.md`.

---

## 1. Executive Summary

This report establishes a rigorous, unified audit of all known limitations, sub-optimal results, regressions, and failures within the NeuralOps platform. In strict accordance with the repository's scientific honesty guidelines, every negative result and hardware bottleneck is documented with exact provenance, correlating theoretical and textual claims in academic papers and documentation against raw empirical evidence in benchmark ledger files.

### Key Audit Findings
1. **The 203-Module Historical Ledger & Release-Gate Discrepancy:**
   - As documented in `CHANGELOG.md` (lines 183–186) and `docs/changelog.md` (lines 46–49) for release v1.5.0, the 203-module unified release gate recorded:
     $$\text{PASS\_FAST}=15, \quad \text{PASS\_SLOW}=115, \quad \text{FAIL\_CORRECTNESS}=71, \quad \text{FAIL}=1, \quad \text{TIMEOUT}=1$$
   - These 71 correctness failures and 1 timeout were explicitly preserved in documentation to prevent misrepresenting unverified modules as accelerated.
2. **Sub-1.0x Speedup & cuBLAS Superiority:**
   - `fused_favor_plus`: Triton implementation yields **0.73x** speedup (0.098 ms Triton vs 0.071 ms PyTorch cuBLAS) on RTX 4070 (SM 8.9), documented in `README.md` Section 16 and verified in `results/triton_all_26_kernels_benchmark.json` (line 165).
   - In `results/triton_full_leaderboard_v136.csv`, multiple Triton kernels and compression engines achieve $<1.0\times$ speedup, including `res_svd` (0.01x), `dynamic_svd` (0.21x), `dobi_svd` (0.24x), `a3_compressor` (0.31x), `linear_attn.py` (0.20x), `low_rank.py` (0.25x), `sparse_attn.py` (0.26x), `state_update.py` (0.60x), and `fused_ce.py` (0.98x).
   - In `results/master_108_cuda_modules_benchmark.csv`, `paged_recurrent_step` exhibits severe latency regression (CUDA 30.35 ms vs Ref 0.037 ms — 826x slower than PyTorch).
3. **Four Pseudo-Triton Functions:**
   - Four low-rank operations in `neuralops/ops/triton/low_rank.py` previously bore `_triton` suffixes despite executing PyTorch matrix multiplications and contractions. These were formally deprecated in v1.8.1 (`CHANGELOG.md` line 112, commit `3b00eeb09a`):
     1. `fused_nystrom_landmark_pinv_step_triton` $\rightarrow$ `fused_nystrom_landmark_pinv_step_pytorch`
     2. `fused_rpc_sketch_attention_triton` $\rightarrow$ `fused_rpc_sketch_attention_pytorch`
     3. `fused_primal_dual_landmark_triton` $\rightarrow$ `fused_primal_dual_landmark_pytorch`
     4. `fused_svd_mamba_proj_triton` $\rightarrow$ `fused_svd_mamba_proj_pytorch`
4. **Hardware Wall & Out-of-Memory (OOM) Limits:**
   - Dense quadratic attention and FlashAttention-2 suffer hard CUDA OOM at sequence length $N \ge 8192$ on 12 GB Ada Lovelace hardware (>22.4 GB required), grounding the theoretical necessity for linear attention and SSMs.
5. **Direct SVD & Naive Cross-Entropy Collapse:**
   - Direct SVD parameter reduction without activation whitening drops DistilBERT SST-2 accuracy to 34.80% and RoBERTa to 36.20% at $r=4$ (worse than random guess, 50%).
   - Post-surgery naive cross-entropy training collapses hidden state cosine similarity to $<0.25$, establishing the mandatory requirement for 6-pillar knowledge distillation.

---

## 2. Comprehensive Bad Results, Failures & Bottlenecks Ledger

The following table correlates every identified failure, bottleneck, or sub-optimal metric from `academic/` and `docs/` with the exact underlying record in `results/`:

| Module / Kernel Name | Subsystem / Category | Claimed Metric (Academic / Docs) | Recorded Metric (Results Data) | Failure Status | Source File & Line Number |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fused_favor_plus` | Triton Kernel / Attention | 0.73x speedup (cuBLAS preferred for $D < 256$) | Triton: 0.098 ms, PyTorch: 0.071 ms, Speedup: 0.73x, Max diff: 6e-6 | `SUB_OPTIMAL` (< 1.0x) | `README.md:1654`, `results/triton_all_26_kernels_benchmark.json:165` |
| 203-Module Ledger Gate | Unified Benchmark Release Gate | 71 correctness failures, 1 timeout, 1 fail out of 203 | $\text{PASS\_FAST}=15$, $\text{PASS\_SLOW}=115$, $\text{FAIL\_CORRECTNESS}=71$, $\text{FAIL}=1$, $\text{TIMEOUT}=1$ | `FAIL_CORRECTNESS` (71 cases), `TIMEOUT` (1 case) | `CHANGELOG.md:183-186`, `docs/changelog.md:46-49` |
| `fused_nystrom_landmark_pinv_step_triton` | Triton Low-Rank Operator | Pseudo-Triton function (called PyTorch matmuls) | Pure PyTorch fallback without native Triton JIT code | `DEPRECATED_PSEUDO_TRITON` | `CHANGELOG.md:112`, `neuralops/ops/triton/low_rank.py:1114` |
| `fused_rpc_sketch_attention_triton` | Triton Low-Rank Operator | Pseudo-Triton function (called PyTorch einsums) | Pure PyTorch fallback without native Triton JIT code | `DEPRECATED_PSEUDO_TRITON` | `CHANGELOG.md:112`, `neuralops/ops/triton/low_rank.py:1154` |
| `fused_primal_dual_landmark_triton` | Triton Low-Rank Operator | Pseudo-Triton function (called PyTorch matmuls) | Pure PyTorch fallback without native Triton JIT code | `DEPRECATED_PSEUDO_TRITON` | `CHANGELOG.md:112`, `neuralops/ops/triton/low_rank.py:1181` |
| `fused_svd_mamba_proj_triton` | Triton Low-Rank Operator | Pseudo-Triton function (called PyTorch matmuls) | Pure PyTorch fallback without native Triton JIT code | `DEPRECATED_PSEUDO_TRITON` | `CHANGELOG.md:112`, `neuralops/ops/triton/low_rank.py:1215` |
| `fused_norm_linear` | CUDA C++ Native Kernel | 225–244x slower than PyTorch (uncoalesced stride, serial loop) | CUDA: 827.59 ms vs PyTorch: 3.39 ms ($B=8, S=512, H=4096$) | `FALLBACK_ONLY` / `SEVERE_SLOWDOWN` | `docs/fallback_list.md:10-21`, `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:98` |
| `fp8_gemm` | CUDA C++ Native Kernel | 0.15x speedup (6.7x slower; lacks Ada Lovelace TMA) | CUDA: 0.507 ms vs PyTorch: 0.078 ms | `SUB_OPTIMAL` (0.15x) | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:99` |
| `fused_cross_entropy` | CUDA C++ Native Kernel | 0.31x speedup (3.2x slower; cuBLAS online softmax saturated) | CUDA: 0.037 ms vs PyTorch: 0.012 ms | `SUB_OPTIMAL` (0.31x) | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:100` |
| `w4a16_gemm` | CUDA C++ Native Kernel | 0.39x speedup (2.5x slower; bit-unpacking register pressure) | CUDA: 0.266 ms vs PyTorch: 0.104 ms | `SUB_OPTIMAL` (0.39x) | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:101` |
| `fused_lowrank_gated_gemm` | CUDA C++ Native Kernel | 0.65x speedup (memory bandwidth overhead) | CUDA: 0.038 ms vs PyTorch: 0.025 ms | `SUB_OPTIMAL` (0.65x) | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:102` |
| `speculative_verify` | CUDA C++ Native Kernel | 0.71x–0.73x speedup (slower than PyTorch baseline) | CUDA: 0.0518 ms, Ref: 0.0 ms, Speedup: 0.53x | `FALLBACK_ONLY` | `docs/fallback_list.md:29-35`, `results/master_108_cuda_modules_benchmark.csv:98` |
| `entropy_pruning` | CUDA Kernel Benchmark | Exceeded 120-second isolated benchmark limit | Execution stopped after timeout | `TIMEOUT / FALLBACK_ONLY` | `docs/fallback_list.md:37-40` |
| `logcumsumexp` | CUDA Kernel Benchmark | Exceeded 120-second isolated benchmark limit | Execution stopped after timeout | `TIMEOUT / FALLBACK_ONLY` | `docs/fallback_list.md:42-45` |
| `samba_moe` | CUDA Kernel Benchmark | Exceeded 120-second isolated benchmark limit | Execution stopped after timeout | `TIMEOUT / FALLBACK_ONLY` | `docs/fallback_list.md:47-50` |
| `sparse_attention` | CUDA Kernel Benchmark | Exceeded 120-second isolated benchmark limit | Execution stopped after timeout | `TIMEOUT / FALLBACK_ONLY` | `docs/fallback_list.md:52-55` |
| `rotational_svd` | CUDA C++ Kernel | Benchmark execution timeout | Empty benchmark output after timeout | `TIMEOUT` | `results/master_108_cuda_modules_benchmark.csv:81` |
| `rpca_sparse_svd` | CUDA C++ Kernel | Benchmark execution timeout | Empty benchmark output after timeout | `TIMEOUT` | `results/master_108_cuda_modules_benchmark.csv:83` |
| `tt_contraction` | CUDA C++ Kernel | Benchmark execution timeout | Empty benchmark output after timeout | `TIMEOUT` | `results/master_108_cuda_modules_benchmark.csv:105` |
| `paged_recurrent_step` | CUDA C++ Kernel | Large numerical divergence and high latency | CUDA: 30.35 ms, Ref: 0.037 ms (826x slower), max_diff: 0.2846 | `FAIL_NUMERICS` | `results/master_108_cuda_modules_benchmark.csv:72` |
| `phantom` | CUDA C++ Kernel | Numerical difference exceeded threshold | max_diff = 0.15234 (> 0.05 budget) | `FAIL_NUMERICS` | `results/master_108_cuda_modules_benchmark.csv:73` |
| `rfa_attention` | CUDA C++ Kernel | Kernel produces NaN values | max_diff = NaN | `FAIL_NAN` | `results/master_108_cuda_modules_benchmark.csv:78` |
| `selective_ssm` | CUDA C++ Kernel | Kernel produces NaN values | max_diff = NaN | `FAIL_NAN` | `results/master_108_cuda_modules_benchmark.csv:88` |
| `spectra_v2` | CUDA C++ Kernel | Numerical difference exceeded threshold | max_diff = 0.4050 (> 0.05 budget) | `FAIL_NUMERICS` | `results/master_108_cuda_modules_benchmark.csv:94` |
| `state_update` | CUDA C++ Kernel | Numerical difference exceeded threshold | max_diff = 0.15625 (> 0.05 budget) | `FAIL_NUMERICS` | `results/master_108_cuda_modules_benchmark.csv:100` |
| `vspec` | CUDA C++ Kernel | Numerical difference exceeded threshold | max_diff = 0.42786 (> 0.05 budget) | `FAIL_NUMERICS` | `results/master_108_cuda_modules_benchmark.csv:109` |
| `aegis_gate_norm` | CUDA C++ Kernel | Process failure during execution | Null output during benchmark run | `CRASH` | `results/master_108_cuda_modules_benchmark.csv:5` |
| `entropy_gate` | CUDA C++ Kernel | Process failure during execution | Null output during benchmark run | `CRASH` | `results/master_108_cuda_modules_benchmark.csv:19` |
| `favor_plus` | CUDA C++ Kernel | Process failure during execution | Null output during benchmark run | `CRASH` | `results/master_108_cuda_modules_benchmark.csv:23` |
| `gla_unnormalized_scan` | CUDA C++ Kernel | Process failure during execution | Null output during benchmark run | `CRASH` | `results/master_108_cuda_modules_benchmark.csv:39` |
| `unified_rope_offset` | CUDA C++ Kernel | Process failure during execution | Null output during benchmark run | `CRASH` | `results/master_108_cuda_modules_benchmark.csv:106` |
| `res_svd` | Compression Engine | Severe Triton latency regression | PyTorch: 0.09 ms, Triton: 10.09 ms, Speedup: 0.01x | `SUB_OPTIMAL` (0.01x) | `results/triton_full_leaderboard_v136.csv:67` |
| `dynamic_svd` | Compression Engine | Sub-1.0x Triton throughput | PyTorch: 0.03 ms, Triton: 0.16 ms, Speedup: 0.21x | `SUB_OPTIMAL` (0.21x) | `results/triton_full_leaderboard_v136.csv:66` |
| `dobi_svd` | Compression Engine | Sub-1.0x Triton throughput | PyTorch: 0.04 ms, Triton: 0.17 ms, Speedup: 0.24x | `SUB_OPTIMAL` (0.24x) | `results/triton_full_leaderboard_v136.csv:65` |
| `a3_compressor` | Compression Engine | Sub-1.0x Triton throughput | PyTorch: 0.05 ms, Triton: 0.16 ms, Speedup: 0.31x | `SUB_OPTIMAL` (0.31x) | `results/triton_full_leaderboard_v136.csv:64` |
| `dora_qlora` | Compression Engine | Sub-1.0x Triton throughput | PyTorch: 0.18 ms, Triton: 0.21 ms, Speedup: 0.84x | `SUB_OPTIMAL` (0.84x) | `results/triton_full_leaderboard_v136.csv:63` |
| `kmeans_quantize` | Compression Engine | Sub-1.0x Triton throughput | PyTorch: 0.24 ms, Triton: 0.27 ms, Speedup: 0.90x | `SUB_OPTIMAL` (0.90x) | `results/triton_full_leaderboard_v136.csv:62` |
| `svd_llm` | Compression Engine | Marginal parity / sub-1.0x | PyTorch: 0.11 ms, Triton: 0.11 ms, Speedup: 0.99x | `SUB_OPTIMAL` (0.99x) | `results/triton_full_leaderboard_v136.csv:61` |
| `linear_attn.py` | Core Triton Kernel | Huge numerical drift on chunk scan | PyTorch: 0.02 ms, Triton: 0.101 ms, Speedup: 0.20x, max_abs_diff: 11488.0 | `SUB_OPTIMAL` / `HIGH_DIFF` | `results/triton_full_leaderboard_v136.csv:27` |
| `low_rank.py` | Core Triton Kernel | Sub-1.0x low-rank SVD forward | PyTorch: 0.02 ms, Triton: 0.079 ms, Speedup: 0.25x | `SUB_OPTIMAL` (0.25x) | `results/triton_full_leaderboard_v136.csv:26` |
| `sparse_attn.py` | Core Triton Kernel | Sub-1.0x LSH hashing | PyTorch: 0.102 ms, Triton: 0.395 ms, Speedup: 0.26x | `SUB_OPTIMAL` (0.26x) | `results/triton_full_leaderboard_v136.csv:25` |
| `state_update.py` | Core Triton Kernel | Sub-1.0x recurrent SSM step | PyTorch: 0.039 ms, Triton: 0.064 ms, Speedup: 0.60x | `SUB_OPTIMAL` (0.60x) | `results/triton_full_leaderboard_v136.csv:24` |
| `fused_ce.py` | Core Triton Kernel | Marginal sub-1.0x cross-entropy | PyTorch: 2.307 ms, Triton: 2.346 ms, Speedup: 0.98x | `SUB_OPTIMAL` (0.98x) | `results/triton_full_leaderboard_v136.csv:23` |
| `minimal_svd` | Attention Architecture | Sub-1.0x attention latency | PyTorch: 1.10 ms, Triton: 1.42 ms, Speedup: 0.77x | `SUB_OPTIMAL` (0.77x) | `results/triton_full_leaderboard_v136.csv:53` |
| `svd_compressed` | Attention Architecture | Sub-1.0x attention latency | PyTorch: 16.77 ms, Triton: 17.15 ms, Speedup: 0.98x | `SUB_OPTIMAL` (0.98x) | `results/triton_full_leaderboard_v136.csv:52` |
| `tt_compressor` | Compressor / Language Modeling | Severe perplexity explosion on WikiText-2 | PPL = 2183.78 ($r=16$), PPL = 3036.90 ($r=256$) | `FAIL_PPL_DIVERGED` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:145-146` |
| `aa_svd_factory` | Compressor / Language Modeling | Perplexity explosion | PPL = 1046.66 | `FAIL_PPL_DIVERGED` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:147` |
| `svd_projection` | Compressor / Language Modeling | Perplexity explosion | PPL = 980.91 | `FAIL_PPL_DIVERGED` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:148` |
| `adasvd_compressor` | Compressor / Language Modeling | Perplexity explosion ($r=128$) | PPL = 864.50 | `FAIL_PPL_DIVERGED` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:149` |
| `svd_linear` | Compressor / Language Modeling | Perplexity explosion | PPL = 738.00 | `FAIL_PPL_DIVERGED` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:150` |
| `Direct SVD` ($r=4$) | Low-Rank Surgery Baseline | Accuracy collapse below random guess (50%) | DistilBERT: 34.80%, RoBERTa: 36.20% | `ACCURACY_COLLAPSE` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:54-58` |
| `Naive Cross-Entropy` | Healing / Fine-Tuning Baseline | Latent cosine similarity collapse (< 0.25) | Cosine sim: 0.1578–0.2286, Mamba zero-shot drops 0.8416 $\rightarrow$ 0.2223 | `MANIFOLD_COLLAPSE` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:183-186` |
| `FlashAttention-2` ($N \ge 8192$) | Scaled Dot-Product Attention | Quadratic VRAM wall, Out-of-Memory on 12 GB GPU | Memory requirement > 22.4 GB $\rightarrow$ CUDA OOM | `CUDA_OOM` | `academic/thesis/DEFENSE_QA.md:420-466`, `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:210` |
| `FP8_E5M2` Matrix Ops | Quantization Precision Format | High relative Frobenius error (>9%), throughput drop | Rel error: 0.0905–0.0915, Throughput: 5.92 TFLOPS (vs 26.04 FP16) | `NUMERICAL_DEGRADATION` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:195-205` |
| `Chunk GLA` ($N=4096, B=4$) | Recurrent Linear Attention | L2 cache thrashing (working set 48 MB > 36 MB L2) | L2 hit rate collapsed from 88.4% to 21.2% | `CACHE_THRASHING` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:215-225` |
| `HLA` | Hardware Linear Attention Kernel | GPU hardware watchdog timeout and thread deadlock | `watchdog timeout / CL_OUT_OF_RESOURCES` (`__syncthreads` in branch) | `HARDWARE_DEADLOCK` | `docs/thesis_agents/NEGATIVE_RESULTS_AND_SCIENTIFIC_FRAMING.md:220-230` |
| `mla_attention` | RoBERTa Evaluation Sweep | Child process crashed without output | Latency: 18.16 s, Accuracy: 0.0% | `CRASH_NO_OUTPUT` | `results/summary.csv:2` |
| `inject_spectral_mamba` | BERT Injection Sweep | Shape unpacking exception | Accuracy: 0.0%, "ValueError: not enough values to unpack" | `FAIL` | `results/summary.csv:6` |
| `inject_hymba_transx` | BERT Surgery Sweep | Parameter explosion violation (+33.9%) | Params: 66.95M $\rightarrow$ 89.69M (+22.73M), Accuracy: 47.10% | `REJECTED_EXPANSION` | `results/summary.csv:9` |
| `inject_infini` | BERT Surgery Sweep | Parameter explosion violation (+1.8%) | Params: 66.95M $\rightarrow$ 68.14M (+1.18M), Accuracy: 42.60% | `REJECTED_EXPANSION` | `results/summary.csv:10` |
| `inject_aegis_v2` | BERT Surgery Sweep | Accuracy below chance threshold (36.90%) | Accuracy: 35.20% | `REJECTED_EXPANSION` / `SUB_CHANCE` | `results/summary.csv:14` |
| `cosformer` | BERT Surgery Sweep | Accuracy below chance threshold (36.90%) | Accuracy: 35.60% | `PASS_LOW` | `results/summary.csv:20` |
| `spectraformer` | BERT Surgery Sweep | Accuracy below chance threshold (36.90%) | Accuracy: 34.90% | `PASS_LOW` | `results/summary.csv:21` |
| `compressed_mha` | BERT Surgery Sweep | Accuracy below chance threshold (36.90%) | Accuracy: 34.00% | `PASS_LOW` | `results/summary.csv:26` |
| `mamba_svd_flash` | Cross-Architecture GLUE Sweep | Benchmark execution timed out | Timed out across BERT, RoBERTa, T5 on SST-2 & MRPC | `TIMEOUT` (6 configurations) | `results/benchmark_summary_v2.csv:2-7` |
| `hybrid_svd` | Compressor Sweep | Execution failed on BERT & RoBERTa | Failure across SST-2 and MRPC benchmarks | `FAIL` (4 configurations) | `results/benchmark_summary_v2.csv:8-11` |
| `vspec` / `vspec_attention` | Attention Sweep | Execution failed across BERT, RoBERTa, T5 | Failure across SST-2 and MRPC benchmarks | `FAIL` (12 configurations) | `results/benchmark_summary_v2.csv:12-23` |
| `samba_m3` | Architecture Sweep | Execution failed across BERT, RoBERTa, T5 | Failure across SST-2 and MRPC benchmarks | `FAIL` (6 configurations) | `results/benchmark_summary_v2.csv:24-29` |
| `jamba_info` | Architecture Sweep | Execution failed across BERT, RoBERTa, T5 | Failure across SST-2 and MRPC benchmarks | `FAIL` (6 configurations) | `results/benchmark_summary_v2.csv:30-35` |
| `hymba_transx` | Architecture Sweep | Execution failed across BERT, RoBERTa, T5 | Failure across SST-2 and MRPC benchmarks | `FAIL` (6 configurations) | `results/benchmark_summary_v2.csv:36-41` |
| `leka_v2` / `leka_v3` | Attention Sweep | Execution failed across BERT, RoBERTa, T5 | Failure across SST-2 and MRPC benchmarks | `FAIL` (12 configurations) | `results/benchmark_summary_v2.csv:42-53` |
| `adasplash_attention` | Attention Sweep | Execution failed across BERT, RoBERTa, T5 | Failure across SST-2 and MRPC benchmarks | `FAIL` (6 configurations) | `results/benchmark_summary_v2.csv:54-59` |
| `fused_svd_kernel` | Kernel Sweep | Execution failed across BERT, RoBERTa, T5 | Failure across SST-2 and MRPC benchmarks | `FAIL` (6 configurations) | `results/benchmark_summary_v2.csv:60-65` |
| `kmeans_quantize` | Compressor Sweep | Execution failed across BERT, RoBERTa, T5 | Failure across SST-2 and MRPC benchmarks | `FAIL` (6 configurations) | `results/benchmark_summary_v2.csv:66-71` |
| `svd_sequence` | Multi-Model Unified Benchmark | Overall module failure | Accuracy: 0.0% across DistilBERT, GPT-2, RoBERTa | `FAIL` | `academic/thesis/master_verified_results.json:13214-13247` |
| `svd_sequence_attention` | Multi-Model Unified Benchmark | Overall module failure | Accuracy: 0.0% across DistilBERT, GPT-2, RoBERTa | `FAIL` | `academic/thesis/master_verified_results.json:13248-13280` |
| `tt_linear` | Multi-Model Unified Benchmark | Failed across multiple architectures | Accuracy: 0.0% across DistilBERT, GPT-2, RoBERTa | `FAIL` | `academic/thesis/master_verified_results.json:13460-13490` |
| `qcqa_attention` | RoBERTa SST-2 Benchmark | Benchmark failed during evaluation | Status: failed | `FAIL` | `results/summaries/failures.md:5` |
| `selective_ssm` | RoBERTa SST-2 Convergence Sweep | Extended convergence run failed | Status: failed | `FAIL` | `results/roberta_extended_convergence/summaries/failures.md:7` |
| `gla_attention` | RoBERTa SST-2 Convergence Sweep | Extended convergence run failed | Status: failed | `FAIL` | `results/roberta_extended_convergence/summaries/failures.md:8` |
| `svd_mamba` | RoBERTa SST-2 Convergence Sweep | Extended convergence run failed | Status: failed | `FAIL` | `results/roberta_extended_convergence/summaries/failures.md:9` |

---

## 3. Detailed Cross-Reference with Root Documentation

### 3.1 README.md Sections 16, 17, and 18

#### Section 16: Honest Performance Claims & Fused Kernel Spectrum
- **Textual Claim (`README.md:1637-1655`):**
  Section 16 explicitly documents that not all Triton kernels outperform cuBLAS across every tensor shape. It partitions the 26 Triton kernel modules into:
  - High-Speedup Kernels (> 2.0x): 13 modules
  - Moderate-Speedup Kernels (1.0x–2.0x): 12 modules
  - Sub-Optimal / cuBLAS Preferred (< 1.0x): 1 module (`fused_favor_plus` at **0.73x**)
- **Underlying Micro-Architectural Rationale:**
  NVIDIA cuBLAS incorporates deeply hand-tuned assembly micro-pipelines. For small dimensions ($D < 256$), Triton JIT compiler launch overhead exceeds the raw arithmetic compute time. The platform implements automatic dispatch routing via `check_triton_suitability()` to redirect sub-256 shapes to PyTorch cuBLAS.
- **Empirical Grounding in `results/`:**
  Directly validated in `results/triton_all_26_kernels_benchmark.json` (line 165), recording 0.098 ms (Triton) vs 0.071 ms (PyTorch cuBLAS), yielding 0.73x speedup on shape `(256x64, M=128)`.

#### Section 17: When NOT to Use NeuralOps
- **Textual Claim (`README.md:1658-1667`):**
  Identifies five operational scenarios where NeuralOps is suboptimal:
  1. *Short Sequences ($T < 512$)*: Standard FlashAttention-2 is faster because the linear $\mathcal{O}(T)$ constant overhead exceeds $512^2$.
  2. *Sub-millisecond batch-1 serving*: Specialized static C++ engines (TensorRT-LLM) outperform dynamic Python/Triton runtimes.
  3. *Pure weight-only quantization without surgery*: Point tools (`AutoAWQ`, `bitsandbytes`) have lighter footprints.
  4. *Massive datacenter pretraining (>70B)*: Megatron-LM/DeepSpeed ZeRO-3 are required.
  5. *Legacy hardware (SM < 6.1)*: Compute architectures lacking Triton support fall back to unaccelerated reference paths.

#### Section 18: Known Limitations & Failure Modes
- **Textual Claim (`README.md:1670-1681`):**
  1. *Unactivated Gaussian Singularity*: Raw Gaussian inputs ($\mathbb{E}[x] = 0$) into `gla_scan.py` cause output explosion if the denominator $q_t Z_t + 10^{-4}$ crosses zero. Mitigated in v2.0.1 by forcing `ELU(x) + 1.0` and hardware denominator clamping (`tl.maximum(denom, 1e-4)`).
  2. *Bidirectional Recurrence in Causal Models*: Bidirectional scans (`0.5 * (y_fwd + y_bwd)`) violate autoregressive causality if applied to GPT-2/LLaMA decoders.
  3. *Small Batch GEMM Launch Overhead*: High launch latency for $B=1, T<128$.

---

### 3.2 CHANGELOG.md Version History Entries

#### Release v1.5.0 Entry (`CHANGELOG.md:183-186` & `docs/changelog.md:46-49`)
- **Explicit Acknowledgment of Correctness Failures:**
  ```markdown
  ### Known Limitations
  - The final 203-module ledger records `PASS_FAST=15`, `PASS_SLOW=115`, `FAIL_CORRECTNESS=71`, `FAIL=1`, `PASS_HIGH_DIFF=0`, and `TIMEOUT=1`.
  - The 71 correctness failures and one timeout remain explicitly documented; they are not treated as successful acceleration.
  ```
- **Correlation with Benchmark Codebase:**
  The 71 correctness failure rows originated in `neuralops/benchmarks/ops_runner.py` (lines 534–541) and `neuralops/benchmarks/verify_ops.py` (lines 53–85), which assign `FAIL_CORRECTNESS` whenever `config.func is None`, `config.ref_func is None`, output contains non-finite values (NaN/Inf), or numerical difference exceeds the threshold budget (`max_diff > atol`).

#### Release v1.8.1 Entry (`CHANGELOG.md:112`)
- **Pseudo-Triton Function Deprecation:**
  ```markdown
  ### Fixed
  - Renamed four pseudo-Triton functions in `low_rank.py` to `*_pytorch`; old names remain as deprecated aliases emitting `DeprecationWarning`.
  ```
- In commit `3b00eeb09a` (`low_rank.py:1114-1230`), `fused_nystrom_landmark_pinv_step_triton`, `fused_rpc_sketch_attention_triton`, `fused_primal_dual_landmark_triton`, and `fused_svd_mamba_proj_triton` were audited as purely PyTorch matrix wrappers. They were safely renamed to `*_pytorch` and configured with transparent backward-compatibility shims emitting explicit `DeprecationWarning`s.

---

## 4. Architectural Synthesis & Methodological Value

Far from undermining the credibility of the NeuralOps framework, these negative results and documented bottlenecks constitute its primary scientific strengths:
1. **Validation of 3-Level Dispatch:** If custom CUDA/Triton kernels were uniformly faster across all operations, the complex dynamic dispatch hierarchy (`CUDA -> Triton -> PyTorch`) would represent unnecessary engineering overhead. Slower kernels in memory-bound reduction regimes (`fused_cross_entropy`, `fused_norm_linear`) prove why dynamic routing to `cuBLAS` is essential.
2. **Justification of HNC-SVD:** The collapse of naive direct SVD ($34.80\%$ accuracy) provides the empirical counterproof establishing why Cholesky activation whitening ($\boldsymbol{\Sigma}_{XX} = \mathbf{L}\mathbf{L}^\top$) and Hessian null-space projection are theoretically required.
3. **Necessity of 6-Pillar Distillation:** The catastrophic loss of manifold structure under hard label cross-entropy ($<0.25$ cosine similarity) provides the empirical justification for Stiefel-manifold and Wasserstein geometry preservation during post-surgery recovery.
4. **Data Integrity Standard:** All 71 correctness failures and empirical timeouts are preserved and cataloged, preventing silent regressions and upholding the zero-fake-artifact standard of the platform.
