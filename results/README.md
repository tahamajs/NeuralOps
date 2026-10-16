# 🏛️ NEURALOPS EXPERIMENT RESULTS LEDGER & PROVENANCE REGISTRY

Welcome to the **NeuralOps (`SVD_linear_Attention` v1.3.7) Experiment Results Ledger & Provenance System**.

This directory serves as the unified, immutable, version-controlled provenance and benchmark tracking engine for all attention mechanisms, compression engines, Triton kernels, CUDA kernels, and end-to-end models in the NeuralOps platform.

---

## 📐 Directory Structure & Architecture

```
results/
├── README.md                          # This document
├── INDEX.md                           # Legacy directory navigation index
├── master_benchmark_summary.csv       # High-level aggregated benchmark metrics across all sweeps
├── master_aggregated_results.csv      # Fine-grained metrics table
├── pareto_table.csv                   # Pareto-optimal latency vs accuracy configurations
├── compatibility_results.csv          # Architecture compatibility matrix
│
├── registry/                          # 🏛️ Canonical Dual-Storage Experiment Ledger
│   ├── master_ledger.jsonl            # Git-tracked append/upsert JSONL records with full provenance
│   ├── master_ledger.sqlite           # High-performance indexed SQLite mirror for instant SQL/CLI queries
│   └── index_summary.json             # High-level ledger coverage, family counts, and latest versions
│
├── by_category/                       # 📂 Categorical Profiles & Leaderboards
│   ├── attentions/                    # 131 Linear, Recurrent, Chunk, and Hybrid Attention mechanisms
│   ├── compressors/                   # 72 SVD, Low-Rank, Activation-Aware, and Quantized Compressors
│   ├── triton_kernels/                # 26 Core + 13 Flagship Triton JIT kernels (SM 8.9 / Ada / Hopper)
│   ├── cuda_kernels/                  # Level 1, 2, 3 Optimized CUDA C++/PyTorch bindings
│   └── ablations_and_kd/              # HNC-SVD nullspace, 6-Pillar KD, and RoBERTa micro-healing ablations
│
├── by_module/                         # 📜 Per-Module Historical Dossiers
│   └── <module_name>.json             # Full lifecycle history: all runs, versions, methods, git SHAs, timestamps
│
├── plots/                             # 📊 Generated Charts & Vector Visualizations
│   ├── latency_vs_accuracy.svg        # Pure vector SVG Pareto frontiers
│   ├── compression_ratio_curve.svg    # Parameter compression vs recovery curves
│   └── throughput_ladder.svg          # Roofline throughput bar charts
│
└── [00_archive ... 12_paper_runs]/    # Raw experimental run folders & sweep artifacts
```

---

## 🛡️ Provenance Invariant & Schema

Every single benchmark record recorded in NeuralOps adheres to the strict **`ResultRecord`** specification:

| Field | Type | Description |
| :--- | :--- | :--- |
| `record_id` | `str` | Unique, deterministic or UUID key (`rec_<md5_hash>`) |
| `module_name` | `str` | Canonical module name matching registry (e.g. `gated_deltanet`, `hnc_svd`) |
| `category` | `str` | Subsystem type (`attention`, `compressor`, `triton_kernel`, `cuda_kernel`, `end_to_end_model`) |
| `family` | `str` | Algorithmic family (e.g. `gated-delta-recurrent`, `activation-aware-svd`, `simplex-wgqa`) |
| `version_tag` | `str` | Package version tag when run occurred (e.g. `v1.3.7`) |
| `revision` | `int` | Sequential revision count per module (1, 2, 3...) |
| `status` | `str` | Record status: `verified`, `superseded`, or `deprecated` |
| `timestamp_utc` | `str` | Exact ISO-8601 UTC timestamp of creation/ingestion |
| `git_commit_sha`| `str` | Exact 10-char or 40-char Git commit SHA corresponding to the code snapshot |
| `generation_method` | `str` | Complete method provenance (e.g. `"Activation-Aware Fisher-Weighted SVD (FWSVD)"`) |
| `experimental_setup` | `dict` | Hardware, dataset, backbone, seq_len, rank, batch size, seed |
| `metrics` | `dict` | Exact evaluated numbers (`eval_accuracy`, `latency_ms`, `compression_ratio`, `tflops`, etc.) |
| `notes` | `str` | Reproducibility notes, convergence logs, or environment quirks |

---

## 💻 CLI Usage (`neuralops results` / `neuralops ledger`)

The ledger is natively built into the `neuralops` CLI.

### 1. Querying Experiments
```bash
# Query all verified results for a module
neuralops results query --module hnc_svd

# Query top results in a category filtered by minimum accuracy
neuralops results query --category attention --min-metric eval_accuracy 90.0

# Filter by Git Commit SHA or method
neuralops results query --git-sha a1b2c3d4e5
neuralops results query --method "HNC Nullspace Projection"
```

### 2. Inspecting Module History & Version Diff
```bash
# Display full historical dossier of an attention or compressor
neuralops results dossier --module gated_deltanet

# Compare two versions of a module to inspect performance delta
neuralops results diff --module roberta_12layer_kd --v1 1 --v2 2
```

### 3. Rendering Charts & Exporting Tables
```bash
# Render ASCII terminal chart of top attention mechanisms
neuralops results plot --category attention --metric eval_accuracy --chart-type bar --format ascii

# Generate high-resolution standalone vector SVG chart
neuralops results plot --category compressor --metric compression_ratio --format svg --out results/plots/compressor_ratios.svg

# Export verified results as LaTeX booktabs table for paper submissions
neuralops results export --category attention --format latex --out paper/iclr2027-submission/tables/attention_benchmarks.tex

# Export as clean Markdown table
neuralops results export --category compressor --format markdown
```

### 4. Adding or Updating Records via CLI
```bash
neuralops results add \
  --module simplex_wgqa \
  --category attention \
  --family simplex-wgqa \
  --method "Simplex-Constrained WGQA + Closed-Form Ridge Calibration" \
  --metrics '{"eval_accuracy": 91.8, "speedup": 2.4}' \
  --dataset sst2 \
  --backbone roberta-base \
  --mode new_version
```

---

## 🐍 Python API Integration

### Direct Logging from Experiments
```python
from neuralops import log_experiment_result

# Automatically logs to JSONL, SQLite, and module dossier with Git SHA provenance
log_experiment_result(
    module_name="hnc_svd",
    category="compressor",
    family="hnc-svd",
    generation_method="HNC Nullspace Projection + Cholesky Whitening",
    experimental_setup={
        "dataset": "sst2",
        "backbone": "roberta-base",
        "rank": 32,
        "device": "NVIDIA RTX 4090",
    },
    metrics={
        "eval_accuracy": 92.4,
        "compression_ratio": 4.0,
        "latency_ms": 2.15,
        "parameter_reduction": 0.75,
    },
    notes="Zero-drop calibration via closed-form Cholesky whitening",
    mode="new_version",  # bumps revision and supersedes prior run
)
```

### Automatic Telemetry in NeuralOps Subsystems

All core NeuralOps engines automatically record their benchmark and diagnostic runs into the ledger:
1. **Hardware Roofline Profiler (`neuralops.core.roofline.profile_roofline`)**:
   Automatically logs arithmetic intensity, latency, and TFLOPS when `log_to_ledger=True`.
2. **Model Surgery Micro-Healing (`neuralops.core.surgery.heal_replaced_model`)**:
   Automatically logs pre-healing vs. post-healing recovery accuracies and speedups.
3. **Training & Distillation Manager (`neuralops.trainer.results_manager.ResultsManager`)**:
   Directly persists cross-epoch validation trajectories into the ledger.
4. **Architecture Inspector (`neuralops inspect <module_name>`)**:
   Automatically loads and renders the verified ledger benchmark card inside the CLI terminal card!
