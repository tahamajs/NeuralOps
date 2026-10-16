# Benchmark Reconciliation Report

**Date**: October 16, 2026  
**Auditor**: Principal Systems Architect, Benchmarking Lead  
**Scope**: Reconciliation between raw JSON execution benchmarks and published Markdown / LaTeX reports.

---

## 1. Executive Summary

All primary benchmark artifacts generated across the benchmark sweeps were cross-referenced against the raw JSON execution logs in `results/`:
1. `results/triton_all_26_kernels_benchmark.json`
2. `results/competitive_comparison_v137.json`
3. `results/hnc_svd_comprehensive_benchmark.json`
4. `results/roberta_glue_compression_results.json`
5. `results/triton_full_leaderboard_v136.json`

**Finding**: The Markdown and LaTeX leaderboards faithfully reflect the raw floating-point measurements recorded during test execution on the NVIDIA RTX GPU hardware.

---

## 2. Triton Kernels vs. PyTorch Eager Reconciliation

| Triton Kernel | Shape / Context | Triton Latency (ms) | PyTorch Eager (ms) | Speedup Ratio | Numerical Max Diff | Validation Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `fused_hnc_nullspace_step` | 2048 x 2048 | 0.459 ms | 3.441 ms | **7.49x** | 0.007812 | :white_check_mark: Verified |
| `sliding_window_attention` | $T=2048, W=128$ | 0.103 ms | 6.821 ms | **65.92x** | 0.001953 | :white_check_mark: Verified |
| `gla_scan` | $T=1024, D=64$ | 0.842 ms | 3.654 ms | **4.34x** | 0.003906 | :white_check_mark: Verified |
| `gdn_scan_triton` | $T=2048, H=8$ | 0.720 ms | 3.250 ms | **4.51x** | 0.001953 | :white_check_mark: Verified |
| `retention_chunk_scan` | $T=4096, C=64$ | 1.120 ms | 5.840 ms | **5.21x** | 0.000977 | :white_check_mark: Verified |

---

## 3. GLUE SST-2 Compression Benchmark Reconciliation

Verified from `results/roberta_glue_compression_results.json` and `results/competitive_comparison_v137.json`:

| Compressor | Rank ($r$) | Baseline Acc (%) | Post-Compression Acc (%) | Post-KD Recov Acc (%) | Compression Ratio | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AdaSVD** | 32 | 93.80% | 79.40% | **80.40%** | 24.0x | :white_check_mark: In-Sync |
| **FWSVD** | 32 | 93.80% | 79.10% | **80.10%** | 24.0x | :white_check_mark: In-Sync |
| **SparseGPT** | 128 | 93.80% | 88.50% | **91.40%** | 6.0x | :white_check_mark: In-Sync |
| **HNC-SVD** | 256 | 93.80% | 46.77% | 46.77% | 3.0x | :warning: High-Rank Only |
| **Naive Truncated SVD**| 32 | 93.80% | 53.20% | 53.20% | 24.0x | :white_check_mark: In-Sync |

---

## 4. Discrepancy Reconciliation Action Items

1. **Table Precision**: All published markdown tables have been updated to display speedups rounded to two decimal places, matching the JSON floating-point outputs.
2. **Missing Files Resolved**: The historical `results/benchmarks/` path was verified to be centralized in `results/`, and all references in docs have been pointed directly to `results/`.
