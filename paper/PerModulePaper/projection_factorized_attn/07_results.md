# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_projection_factorized_attn/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 2 --> |
| results/unified_projection_factorized_attn/distilbert/summary.csv | distilbert | projection_factorized_attn | PASS | 81.19% <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 3 --> | 0.8229 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 3 --> |
| results/unified_projection_factorized_attn/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 2 --> |
| results/unified_projection_factorized_attn/gpt2/summary.csv | gpt2 | projection_factorized_attn | PASS_LOW | 54.82% <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 3 --> | 0.6893 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 3 --> |
| results/unified_projection_factorized_attn/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 2 --> |
| results/unified_projection_factorized_attn/roberta/summary.csv | roberta | projection_factorized_attn | PASS | 62.73% <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 3 --> | 0.7043 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_projection_factorized_attn/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 2 --> |
| results/unified_projection_factorized_attn/distilbert/summary.csv | distilbert | projection_factorized_attn | 66955010 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 3 --> | 54624074 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 3 --> | 645192 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 3 --> | 1.2257x <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 3 --> |
| results/unified_projection_factorized_attn/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 2 --> |
| results/unified_projection_factorized_attn/gpt2/summary.csv | gpt2 | projection_factorized_attn | 124441344 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 3 --> | 99779472 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 3 --> | 1290384 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 3 --> | 1.2472x <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 3 --> |
| results/unified_projection_factorized_attn/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 2 --> |
| results/unified_projection_factorized_attn/roberta/summary.csv | roberta | projection_factorized_attn | 124647170 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 3 --> | 107072402 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 3 --> | 1290384 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 3 --> | 1.1641x <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_projection_factorized_attn/distilbert/summary.csv | distilbert | full | 54621.22 tok/s <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 2 --> |
| results/unified_projection_factorized_attn/distilbert/summary.csv | distilbert | projection_factorized_attn | 22010.58 tok/s <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 3 --> | 270.09 MB <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_projection_factorized_attn/distilbert/summary.csv line 3 --> |
| results/unified_projection_factorized_attn/gpt2/summary.csv | gpt2 | full | 22795.14 tok/s <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 2 --> |
| results/unified_projection_factorized_attn/gpt2/summary.csv | gpt2 | projection_factorized_attn | 10065.69 tok/s <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 3 --> | 468.08 MB <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_projection_factorized_attn/gpt2/summary.csv line 3 --> |
| results/unified_projection_factorized_attn/roberta/summary.csv | roberta | full | 23835.57 tok/s <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 2 --> |
| results/unified_projection_factorized_attn/roberta/summary.csv | roberta | projection_factorized_attn | 10877.86 tok/s <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 3 --> | 555.07 MB <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_projection_factorized_attn/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=94.50%, comp=1.065x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.50% | `results/rank_ablation/ablation_all_rows.csv` |
| F1 Score | 0.9414 | `results/rank_ablation/ablation_all_rows.csv` |
| Compression | 1.065x | `results/rank_ablation/ablation_all_rows.csv` |
| Compressed Layers | 6 | `results/rank_ablation/ablation_all_rows.csv` |
| Fallback Layers | 0 | `results/rank_ablation/ablation_all_rows.csv` |
| Timestamp | 2026-09-20T00:37:53.951332 | `results/rank_ablation/ablation_all_rows.csv` |

