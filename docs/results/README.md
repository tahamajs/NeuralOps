---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
language:
- en
license: mit
multilinguality:
- monolingual
size_categories:
- 100K<n<1M
source_datasets:
- original
task_categories:
- text-generation
- text-classification
task_ids:
- language-modeling
pretty_name: NeuralOps Benchmark Results & Empirical Ledger
tags:
- neuralops
- model-surgery
- attention-mechanisms
- compression
- llm-efficiency
- triton
- cuda
- benchmark-results
configs:
- config_name: default
  data_files:
  - split: train
    path: "results.parquet"
---

<!-- Author: Taha Majlesi -->
# NeuralOps Benchmark Results & Artifacts Repository

**Author:** Taha Majlesi
**Release Version:** 1.4.0
**Hardware Environment:** NVIDIA GeForce RTX 4070 (AD104, 12GB VRAM, Compute Capability 8.9)
**Software Stack:** PyTorch 2.5+, Triton 3.0+, CUDA 12.4+

---

## 1. Overview

This directory contains the canonical benchmark results, empirical evaluation matrices, hardware profiling traces, and verification logs for the **NeuralOps** library at release version `v1.4.0`.

All evaluation data points reflect real physical hardware executions under identical reproducible seeds and thermal conditions. The directory is sanitized to maintain zero build clutter (`.err`, `.log`, `.tmp`, and zero-byte files are barred).

---

## 2. Canonical Benchmark Artifacts

The following canonical benchmark files represent the single source of truth for all published metrics in the thesis and research paper:

| File | Format | Description |
| :--- | :---: | :--- |
| `MASTER_RESULTS_MATRIX.csv` | CSV | Master results matrix spanning all attention mechanisms and compressors, recording latency, memory, throughput, and GLUE downstream accuracies. |
| `MASTER_RESULTS_MATRIX.md` | Markdown | Rendered markdown summary table of the global master results matrix. |
| `triton_all_26_kernels_benchmark.json` | JSON | Hardware benchmark report for all 26 Triton kernel modules, measuring speedup vs. native PyTorch baselines. |
| `master_108_cuda_modules_benchmark.json` | JSON | Microbenchmark measurements and speedup figures for all native CUDA modules. |
| `master_108_cuda_modules_benchmark.csv` | CSV | Tabular counterpart of the 108 CUDA modules microbenchmark report. |
| `master_paper_categorized_inventory.json` | JSON | Structured architectural taxonomy and mathematical classifications for all modules. |
| `master_paper_categorized_inventory.csv` | CSV | Tabular taxonomy inventory for paper generation and appendices. |
| `triton_all_131_attentions_benchmark.json` | JSON | Empirical validation and latency metrics for all 131 canonical subquadratic attention mechanisms. |
| `triton_all_72_compressors_benchmark.json` | JSON | Empirical compression metrics, recovery scores, and memory footprints for all 72 compression engines. |
| `SEQLEN_SCALING_MASTER.csv` | CSV | Throughput and memory consumption scaled from sequence lengths $T = 512$ up to $T = 131,072$. |
| `HARDWARE_ENVIRONMENT_MANIFEST.json` | JSON | Full snapshot of hardware topology, GPU compute capability, clocks, and software toolchains. |
| `RELEASE_CHECKSUMS_v1.4.0.txt` | Text | SHA-256 cryptographic checksums for all canonical result artifacts. |
| `RESULTS_MASTER_INDEX.json` | JSON | Master JSON index indexing all experiment categories and suite paths. |
| `INDEX.md` | Markdown | Detailed browsable index across all 268 evaluation suites. |

---

## 3. Directory Layout

The benchmark repository is partitioned into modular directories:

```
results/
├── README.md                                  # This document
├── INDEX.md                                   # Master suite index
├── RESULTS_MASTER_INDEX.json                  # Machine-readable suite index
├── RELEASE_CHECKSUMS_v1.4.0.txt               # Cryptographic integrity manifest
├── HARDWARE_ENVIRONMENT_MANIFEST.json         # Hardware & environment specification
│
├── [Canonical Matrices]
│   ├── MASTER_RESULTS_MATRIX.csv              # Global cross-evaluation matrix
│   ├── master_108_cuda_modules_benchmark.json # 108 CUDA kernels benchmark
│   ├── triton_all_26_kernels_benchmark.json   # 26 Triton kernels benchmark
│   ├── triton_all_131_attentions_benchmark.json# 131 Attentions benchmark
│   └── triton_all_72_compressors_benchmark.json# 72 Compressors benchmark
│
├── plots/                                     # Publication-grade figures & curves
├── paper_figures/                             # Rendered vector diagrams for paper/thesis
├── profiling/                                 # Kernel traces, roofline analysis & Nsight profiles
├── kd_runs/                                   # Knowledge distillation fine-tuning logs & weights
├── roberta_benchmark_sweep/                   # RoBERTa GLUE downstream evaluation sweeps
└── 00_archive/ ... 12_archive/                # Preserved historical experimental suites
```

---

## 4. Data Hygiene and Integrity Assurance

1. **Zero Clutter**: Ephemeral `.log`, `.err`, `.tmp`, and zero-byte files are automatically pruned by hygiene scripts.
2. **Reproducibility**: Experiments are tagged with exact git commit SHAs in `MASTER_RESULTS_MATRIX.csv`.
3. **Cryptographic Validation**: Any modified benchmark file must have its SHA-256 updated in `RELEASE_CHECKSUMS_v1.4.0.txt`.
