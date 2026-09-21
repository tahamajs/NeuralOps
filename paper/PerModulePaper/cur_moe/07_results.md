# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_cur_moe/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_cur_moe/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 2 --> |
| results/unified_cur_moe/distilbert/summary.csv | distilbert | cur_moe | PASS | 74.43% <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> | 0.7824 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> |
| results/unified_cur_moe/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_cur_moe/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 2 --> |
| results/unified_cur_moe/gpt2/summary.csv | gpt2 | cur_moe | PASS_LOW | 50.92% <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> |
| results/unified_cur_moe/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_cur_moe/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_cur_moe/roberta/summary.csv line 2 --> |
| results/unified_cur_moe/roberta/summary.csv | roberta | cur_moe | PASS_LOW | 50.92% <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cur_moe | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_cur_moe/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur_moe/distilbert/summary.csv line 2 --> |
| results/unified_cur_moe/distilbert/summary.csv | distilbert | cur_moe | 66955010 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> | 48496398 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> | 514060 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> | 1.3806x <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> |
| results/unified_cur_moe/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur_moe/gpt2/summary.csv line 2 --> |
| results/unified_cur_moe/gpt2/summary.csv | gpt2 | cur_moe | 124441344 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> | 88502040 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> | 1145880 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> | 1.4061x <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> |
| results/unified_cur_moe/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_cur_moe/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_cur_moe/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_cur_moe/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur_moe/roberta/summary.csv line 2 --> |
| results/unified_cur_moe/roberta/summary.csv | roberta | cur_moe | 124647170 <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> | 85538586 <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> | 802840 <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> | 1.4572x <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cur_moe | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_cur_moe/distilbert/summary.csv | distilbert | full | 51549.23 tok/s <!-- source: results/unified_cur_moe/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_cur_moe/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 2 --> |
| results/unified_cur_moe/distilbert/summary.csv | distilbert | cur_moe | 40708.95 tok/s <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> | 253.68 MB <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> |
| results/unified_cur_moe/gpt2/summary.csv | gpt2 | full | 27140.95 tok/s <!-- source: results/unified_cur_moe/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_cur_moe/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 2 --> |
| results/unified_cur_moe/gpt2/summary.csv | gpt2 | cur_moe | 21553.95 tok/s <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> | 625.48 MB <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> |
| results/unified_cur_moe/roberta/summary.csv | roberta | full | 26781.23 tok/s <!-- source: results/unified_cur_moe/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_cur_moe/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur_moe/roberta/summary.csv line 2 --> |
| results/unified_cur_moe/roberta/summary.csv | roberta | cur_moe | 20364.41 tok/s <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> | 615.57 MB <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cur_moe | Elapsed 0.166s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=94.30%, comp=1.056x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.30% | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| F1 Score | 0.9392 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compression | 1.056x | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Timestamp | 2026-09-19T16:14:33.169476 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |

