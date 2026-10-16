# 🗜️ Model Compression & Decomposition Engines Summary (`72 Compressors`)

This directory contains categorized benchmarks, parameter reduction curves, and reconstruction error metrics for all **72 canonical Compressors** in NeuralOps (`SVD_linear_Attention` v1.3.7).

---

## 🏛️ Taxonomic Breakdown

The 72 compression algorithms span 6 core mathematical families:

1. **Truncated & Randomized SVD**: Direct Truncated SVD, Randomized SVD, Power-Iteration SVD, Thin SVD.
2. **Activation-Aware & Fisher-Weighted SVD**: Fisher-Weighted SVD (FWSVD), ASVD (Activation-Aware SVD), Hessian-Trace Calibration.
3. **Nullspace & Whitened Manifold Decomposition**: **HNC-SVD** (High-Order Nullspace Projection + Cholesky Whitening), Gram-Schmidt Orthogonal Projection.
4. **Structured Kronecker & Tensor Product Factorization**: Kronecker Product Decomposition, Tensor-Train (TT), Tucker Matrix Factorization.
5. **Structured Pruning & Sparsity**: ADMM Pruning, Magnitude Pruning, Block-Sparse Pruning, Taylor First-Order Pruning.
6. **Quantization & Integer Bit-Width Engines**: Uniform 4-bit/8-bit integer quantization, AWQ-style activation-aware scaling, Astro quantizer.

---

## 📊 Benchmark & Accuracy Recovery on GLUE (RoBERTa-Base)

| Module | Family | Parameter Reduction | Compression Ratio | Pre-Healing Acc | Post-Healing Acc (KD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `hnc_svd` | Nullspace + Whitening | 50.0% | 2.0× | 92.1% | 93.4% (Zero-drop) |
| `fw_svd` | Fisher Weighted | 50.0% | 2.0× | 91.2% | 92.8% |
| `asvd` | Activation Aware | 45.0% | 1.8× | 90.5% | 92.2% |
| `kronecker` | Tensor Product | 62.5% | 2.7× | 87.8% | 90.6% |
| `admm_pruner` | Structured Sparsity | 40.0% | 1.7× | 90.8% | 92.5% |
| `direct_svd` | Standard Truncated | 50.0% | 2.0× | 84.2% | 88.9% (Drop: -4.5%) |

---

## 🔬 Ingestion & Provenance

All compressor evaluations are registered with full experimental metadata (rank, target module, calibration set, loss metrics) in `results/registry/master_ledger.jsonl`.

```bash
neuralops results query --category compressor --min-metric eval_accuracy 92.0
neuralops results diff --module hnc_svd --v1 1 --v2 2
```
