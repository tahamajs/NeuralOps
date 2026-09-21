# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_hybrid_svd_layer/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/distilbert/summary.csv | distilbert | hybrid_svd_layer | PASS | 77.18% <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 3 --> | 0.7715 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 3 --> |
| results/unified_hybrid_svd_layer/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/gpt2/summary.csv | gpt2 | hybrid_svd_layer | PASS_LOW | 51.15% <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 3 --> |
| results/unified_hybrid_svd_layer/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/qwen3_5/summary.csv | qwen3_5 | hybrid_svd_layer | FAIL | 0.0% <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 3 --> | 0.557 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 3 --> |
| results/unified_hybrid_svd_layer/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/roberta/summary.csv | roberta | hybrid_svd_layer | PASS_LOW | 50.92% <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_hybrid_svd_layer/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/distilbert/summary.csv | distilbert | hybrid_svd_layer | 66955010 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 3 --> | 35104550 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 3 --> | 43814 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 3 --> | 1.9073x <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 3 --> |
| results/unified_hybrid_svd_layer/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/gpt2/summary.csv | gpt2 | hybrid_svd_layer | 124441344 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 3 --> | 58381104 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 3 --> | 84528 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 3 --> | 2.1315x <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 3 --> |
| results/unified_hybrid_svd_layer/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/qwen3_5/summary.csv | qwen3_5 | hybrid_svd_layer | 763217728 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 3 --> | 0 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 3 --> |
| results/unified_hybrid_svd_layer/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/roberta/summary.csv | roberta | hybrid_svd_layer | 124647170 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 3 --> | 60946250 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 3 --> | 85322 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 3 --> | 2.0452x <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_hybrid_svd_layer/distilbert/summary.csv | distilbert | full | 181098.85 tok/s <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/distilbert/summary.csv | distilbert | hybrid_svd_layer | 155604.92 tok/s <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 3 --> | 258.7 MB <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 3 --> | 0.6618792232539918 <!-- source: results/unified_hybrid_svd_layer/distilbert/summary.csv line 3 --> |
| results/unified_hybrid_svd_layer/gpt2/summary.csv | gpt2 | full | 71741.09 tok/s <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/gpt2/summary.csv | gpt2 | hybrid_svd_layer | 67161.52 tok/s <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 3 --> | 619.22 MB <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 3 --> | 0.70402240815262 <!-- source: results/unified_hybrid_svd_layer/gpt2/summary.csv line 3 --> |
| results/unified_hybrid_svd_layer/qwen3_5/summary.csv | qwen3_5 | full | 1292.52 tok/s <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/qwen3_5/summary.csv | qwen3_5 | hybrid_svd_layer | 1292.52 tok/s <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 3 --> | 1796.78 MB <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hybrid_svd_layer/qwen3_5/summary.csv line 3 --> |
| results/unified_hybrid_svd_layer/roberta/summary.csv | roberta | full | 94421.41 tok/s <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 2 --> |
| results/unified_hybrid_svd_layer/roberta/summary.csv | roberta | hybrid_svd_layer | 81996.21 tok/s <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 3 --> | 630.13 MB <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 3 --> | 0.6672518915600247 <!-- source: results/unified_hybrid_svd_layer/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=77.18%, comp=1.907x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 77.18% | `results/unified_hybrid_svd_layer/distilbert/summary.csv` |
| F1 Score | 0.7715 | `results/unified_hybrid_svd_layer/distilbert/summary.csv` |
| Compression | 1.907x | `results/unified_hybrid_svd_layer/distilbert/summary.csv` |
| Compressed Layers | 36 | `results/unified_hybrid_svd_layer/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_hybrid_svd_layer/distilbert/summary.csv` |
| Timestamp | 2026-09-11T19:09:00.956833 | `results/unified_hybrid_svd_layer/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.94, 57.91, 67.89]
> These numbers must NOT be cited in the final paper.

