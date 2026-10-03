Module neuralops.evaluation.run_baseline_comparison
===================================================
Baseline Comparison Runner: Benchmarks key literature baselines against Proposed method.

Baselines evaluated:
  1. Dense (Original uncompressed baseline model)
  2. Naive SVD (Truncated SVD without data calibration or whitening)
  3. ASVD (Activation-aware SVD scaling, Yuan et al., 2023)
  4. FWSVD / GFWS (Fisher-weighted SVD, Wang et al., 2022)
  5. SVD-LLM (Cholesky activation-whitened SVD, Wang et al., 2024)
  6. Ours (A3 / Adaptive Activation-Aware Compression)

Datasets evaluated:
  - SST-2 (GLUE binary classification)
  - MNLI (GLUE 3-way natural language inference)
  - AG News (4-class topic categorization)

Usage:
    python run_baseline_comparison.py --dry_run
    python run_baseline_comparison.py --rank 64

Functions
---------

`main()`
:   

`parse_args()`
: