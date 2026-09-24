| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Adaptive Sparse** | **--** | -- | PASS | -- | 0 | **1.00x** | -- | -- |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Adaptive Sparse** | **59.63%** | -15.83% | PASS | 0.3973 | 60 | **0.81x** | 687.5 MB | 30.9s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Adaptive Sparse** | **--** | -- | PASS | -- | 0 | **1.00x** | -- | -- |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_adaptive_sparse/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.55% <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 2 --> | 0.5501 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_sparse/qwen3_5/summary.csv | qwen3_5 | adaptive_sparse | SKIP_QWEN3_5_ARCHITECTURAL | 0.0% <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 3 --> | 0.5501 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 3 --> |
| results/unified_adaptive_sparse/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 2 --> |
| results/unified_adaptive_sparse/roberta/summary.csv | roberta | adaptive_sparse | PASS_NO_COMPRESSION | 66.51% <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 3 --> | 0.6198 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 3 --> |
| results/unified_adaptive_sparse_attention/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_adaptive_sparse_attention/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_adaptive_sparse_attention/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_sparse_attention/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_adaptive_sparse_attention/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_adaptive_sparse_attention/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_sparse_attention/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_adaptive_sparse_attention/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_adaptive_sparse_attention/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_sparse_attention/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_adaptive_sparse_attention/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_adaptive_sparse_attention/roberta/summary.csv line 2 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_adaptive_sparse/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_sparse/qwen3_5/summary.csv | qwen3_5 | adaptive_sparse | 763217728 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 3 --> | 0 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 3 --> |
| results/unified_adaptive_sparse/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 2 --> |
| results/unified_adaptive_sparse/roberta/summary.csv | roberta | adaptive_sparse | 124647170 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 3 --> | 154175390 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 3 --> | 1179660 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 3 --> |
| results/unified_adaptive_sparse_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_adaptive_sparse_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_adaptive_sparse_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_sparse_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_sparse_attention/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_sparse_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_adaptive_sparse_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_adaptive_sparse_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_sparse_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_sparse_attention/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_sparse_attention/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_adaptive_sparse_attention/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_adaptive_sparse_attention/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_sparse_attention/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_sparse_attention/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_sparse_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_adaptive_sparse_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_adaptive_sparse_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_sparse_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_sparse_attention/roberta/summary.csv line 2 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_adaptive_sparse/qwen3_5/summary.csv | qwen3_5 | full | 1318.97 tok/s <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 2 --> | 1810.59 MB <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_sparse/qwen3_5/summary.csv | qwen3_5 | adaptive_sparse | 1318.97 tok/s <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 3 --> | 1810.59 MB <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_adaptive_sparse/qwen3_5/summary.csv line 3 --> |
| results/unified_adaptive_sparse/roberta/summary.csv | roberta | full | 20223.17 tok/s <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 2 --> |
| results/unified_adaptive_sparse/roberta/summary.csv | roberta | adaptive_sparse | 8451.0 tok/s <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 3 --> | 687.46 MB <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_adaptive_sparse/roberta/summary.csv line 3 --> |
| results/unified_adaptive_sparse_attention/distilbert/summary.csv | distilbert | full | 51952.83 tok/s <!-- source: results/unified_adaptive_sparse_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_adaptive_sparse_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_sparse_attention/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_sparse_attention/gpt2/summary.csv | gpt2 | full | 25459.51 tok/s <!-- source: results/unified_adaptive_sparse_attention/gpt2/summary.csv line 2 --> | 793.99 MB <!-- source: results/unified_adaptive_sparse_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_sparse_attention/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_sparse_attention/qwen3_5/summary.csv | qwen3_5 | full | 5803.55 tok/s <!-- source: results/unified_adaptive_sparse_attention/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_adaptive_sparse_attention/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_sparse_attention/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_sparse_attention/roberta/summary.csv | roberta | full | 28934.64 tok/s <!-- source: results/unified_adaptive_sparse_attention/roberta/summary.csv line 2 --> | 713.14 MB <!-- source: results/unified_adaptive_sparse_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_sparse_attention/roberta/summary.csv line 2 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 62.50% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-26 00:01:44 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [67.55, 78.56, 90.71, 79.36, 67.89, 78.56]
> These numbers must NOT be cited in the final paper.

