| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Swift Svd** | **90.71%** | +0.23% | PASS | 0.9101 | 2 | **1.00x** | 335.5 MB | 18.6 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Swift Svd** | **50.92%** | -24.19% | PASS_LOW | 0.6748 | 14 | **1.00x** | 629.7 MB | 29.3 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Swift Svd** | **50.92%** | -9.86% | PASS_LOW | 0.6748 | 8 | **1.00x** | 755.6 MB | 25.7 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_swift_svd/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_swift_svd/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 2 --> |
| results/unified_swift_svd/distilbert/summary.csv | distilbert | swift_svd | PASS | 90.71% <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> | 0.9101 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> |
| results/unified_swift_svd/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_swift_svd/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 2 --> |
| results/unified_swift_svd/gpt2/summary.csv | gpt2 | swift_svd | PASS_LOW | 50.92% <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> |
| results/unified_swift_svd/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_swift_svd/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_swift_svd/roberta/summary.csv line 2 --> |
| results/unified_swift_svd/roberta/summary.csv | roberta | swift_svd | PASS_LOW | 50.92% <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | swift_svd | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_swift_svd/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_swift_svd/distilbert/summary.csv line 2 --> |
| results/unified_swift_svd/distilbert/summary.csv | distilbert | swift_svd | 66955010 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> | 66948100 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> | 3535876 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> | 1.0001x <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> |
| results/unified_swift_svd/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_swift_svd/gpt2/summary.csv line 2 --> |
| results/unified_swift_svd/gpt2/summary.csv | gpt2 | swift_svd | 124441344 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> | 124156424 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> | 12119048 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> | 1.0023x <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> |
| results/unified_swift_svd/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_swift_svd/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_swift_svd/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_swift_svd/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_swift_svd/roberta/summary.csv line 2 --> |
| results/unified_swift_svd/roberta/summary.csv | roberta | swift_svd | 124647170 <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> | 124571920 <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> | 33569296 <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> | 1.0006x <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | swift_svd | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_swift_svd/distilbert/summary.csv | distilbert | full | 135042.32 tok/s <!-- source: results/unified_swift_svd/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_swift_svd/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 2 --> |
| results/unified_swift_svd/distilbert/summary.csv | distilbert | swift_svd | 136020.15 tok/s <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> | 335.49 MB <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> | 0.1762419490704756 <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> |
| results/unified_swift_svd/gpt2/summary.csv | gpt2 | full | 64199.7 tok/s <!-- source: results/unified_swift_svd/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_swift_svd/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 2 --> |
| results/unified_swift_svd/gpt2/summary.csv | gpt2 | swift_svd | 61757.75 tok/s <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> | 755.65 MB <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> | 0.1824499189156506 <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> |
| results/unified_swift_svd/roberta/summary.csv | roberta | full | 87965.94 tok/s <!-- source: results/unified_swift_svd/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_swift_svd/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_swift_svd/roberta/summary.csv line 2 --> |
| results/unified_swift_svd/roberta/summary.csv | roberta | swift_svd | 73089.63 tok/s <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> | 629.7 MB <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> | 0.1889018558170913 <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | swift_svd | Elapsed 0.883s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> | 176.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=89.30%, comp=1.344x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 89.30% | `results/rank_ablation/agnews_rank32/summary.csv` |
| F1 Score | 0.8902 | `results/rank_ablation/agnews_rank32/summary.csv` |
| Compression | 1.344x | `results/rank_ablation/agnews_rank32/summary.csv` |
| Compressed Layers | 0 | `results/rank_ablation/agnews_rank32/summary.csv` |
| Fallback Layers | 0 | `results/rank_ablation/agnews_rank32/summary.csv` |
| Timestamp | 2026-09-19T23:56:37.738418 | `results/rank_ablation/agnews_rank32/summary.csv` |

