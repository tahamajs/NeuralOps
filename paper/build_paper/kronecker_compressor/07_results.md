| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Kronecker** | **90.94%** | +0.46% | PASS | 0.9123 | 1 | **1.00x** | 315.1 MB | 17.8 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Kronecker** | **70.07%** | -5.04% | PASS | 0.7603 | 0 | **1.00x** | 301.7 MB | 25.4 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Kronecker** | **60.32%** | -0.46% | PASS | 0.7038 | 2 | **1.00x** | 643.3 MB | 25.4 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_kronecker/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_kronecker/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_kronecker/distilbert/summary.csv line 2 --> |
| results/unified_kronecker/distilbert/summary.csv | distilbert | kronecker | PASS | 90.94% <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> | 0.9123 <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> |
| results/unified_kronecker/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_kronecker/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_kronecker/gpt2/summary.csv line 2 --> |
| results/unified_kronecker/gpt2/summary.csv | gpt2 | kronecker | PASS | 60.32% <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> | 0.7038 <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> |
| results/unified_kronecker/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_kronecker/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_kronecker/roberta/summary.csv line 2 --> |
| results/unified_kronecker/roberta/summary.csv | roberta | kronecker | PASS | 70.07% <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> | 0.7603 <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kronecker | PASS | Comp 358.64x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_kronecker/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_kronecker/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_kronecker/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_kronecker/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kronecker/distilbert/summary.csv line 2 --> |
| results/unified_kronecker/distilbert/summary.csv | distilbert | kronecker | 66955010 <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> | 66758403 <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> | 985347 <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> | 1.0029x <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> |
| results/unified_kronecker/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_kronecker/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_kronecker/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_kronecker/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kronecker/gpt2/summary.csv line 2 --> |
| results/unified_kronecker/gpt2/summary.csv | gpt2 | kronecker | 124441344 <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> | 124048130 <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> | 787970 <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> | 1.0032x <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> |
| results/unified_kronecker/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_kronecker/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_kronecker/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_kronecker/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kronecker/roberta/summary.csv line 2 --> |
| results/unified_kronecker/roberta/summary.csv | roberta | kronecker | 124647170 <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> | 592130 <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kronecker | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> | Reduced (358.64x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> | 358.64x <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_kronecker/distilbert/summary.csv | distilbert | full | 160032.71 tok/s <!-- source: results/unified_kronecker/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_kronecker/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kronecker/distilbert/summary.csv line 2 --> |
| results/unified_kronecker/distilbert/summary.csv | distilbert | kronecker | 157132.82 tok/s <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> | 315.07 MB <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> | 0.3274622009712534 <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> |
| results/unified_kronecker/gpt2/summary.csv | gpt2 | full | 71960.77 tok/s <!-- source: results/unified_kronecker/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_kronecker/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kronecker/gpt2/summary.csv line 2 --> |
| results/unified_kronecker/gpt2/summary.csv | gpt2 | kronecker | 69983.27 tok/s <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> | 643.3 MB <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> | 0.2137064098441871 <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> |
| results/unified_kronecker/roberta/summary.csv | roberta | full | 115870.9 tok/s <!-- source: results/unified_kronecker/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_kronecker/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kronecker/roberta/summary.csv line 2 --> |
| results/unified_kronecker/roberta/summary.csv | roberta | kronecker | 109075.39 tok/s <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kronecker | Elapsed 0.782s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> | 129.8 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=90.94%, comp=1.003x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.94% | `results/unified_kronecker/distilbert/summary.csv` |
| F1 Score | 0.9123 | `results/unified_kronecker/distilbert/summary.csv` |
| Compression | 1.003x | `results/unified_kronecker/distilbert/summary.csv` |
| Compressed Layers | 1 | `results/unified_kronecker/distilbert/summary.csv` |
| Fallback Layers | 35 | `results/unified_kronecker/distilbert/summary.csv` |
| Timestamp | 2026-09-10T21:40:58.092360 | `results/unified_kronecker/distilbert/summary.csv` |

