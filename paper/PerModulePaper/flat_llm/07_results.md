# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_flat_llm/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_flat_llm/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 2 --> |
| results/unified_flat_llm/distilbert/summary.csv | distilbert | flat_llm | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> |
| results/unified_flat_llm/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_flat_llm/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 2 --> |
| results/unified_flat_llm/gpt2/summary.csv | gpt2 | flat_llm | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> |
| results/unified_flat_llm/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_flat_llm/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_flat_llm/roberta/summary.csv line 2 --> |
| results/unified_flat_llm/roberta/summary.csv | roberta | flat_llm | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> | 0.8124 <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | flat_llm | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_flat_llm/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_flat_llm/distilbert/summary.csv line 2 --> |
| results/unified_flat_llm/distilbert/summary.csv | distilbert | flat_llm | 66955010 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> |
| results/unified_flat_llm/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_flat_llm/gpt2/summary.csv line 2 --> |
| results/unified_flat_llm/gpt2/summary.csv | gpt2 | flat_llm | 124441344 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> |
| results/unified_flat_llm/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_flat_llm/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_flat_llm/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_flat_llm/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_flat_llm/roberta/summary.csv line 2 --> |
| results/unified_flat_llm/roberta/summary.csv | roberta | flat_llm | 124647170 <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | flat_llm | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_flat_llm/distilbert/summary.csv | distilbert | full | 53606.92 tok/s <!-- source: results/unified_flat_llm/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_flat_llm/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 2 --> |
| results/unified_flat_llm/distilbert/summary.csv | distilbert | flat_llm | 53197.92 tok/s <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> |
| results/unified_flat_llm/gpt2/summary.csv | gpt2 | full | 27570.33 tok/s <!-- source: results/unified_flat_llm/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_flat_llm/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 2 --> |
| results/unified_flat_llm/gpt2/summary.csv | gpt2 | flat_llm | 27417.97 tok/s <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> | 630.16 MB <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> |
| results/unified_flat_llm/roberta/summary.csv | roberta | full | 27816.89 tok/s <!-- source: results/unified_flat_llm/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_flat_llm/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_flat_llm/roberta/summary.csv line 2 --> |
| results/unified_flat_llm/roberta/summary.csv | roberta | flat_llm | 28454.01 tok/s <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | flat_llm | Elapsed 0.175s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.70% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.9432 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 36 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T16:23:28.181964 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |

