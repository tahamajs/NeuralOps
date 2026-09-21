# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_aegis_v6/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v6/distilbert/summary.csv | distilbert | aegis_v6 | PASS | 87.04% <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 0.8791 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v6/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v6/gpt2/summary.csv | gpt2 | aegis_v6 | PASS_LOW | 51.61% <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 0.6713 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v6/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_aegis_v6/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 2 --> |
| results/unified_aegis_v6/roberta/summary.csv | roberta | aegis_v6 | PASS | 59.98% <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 0.6161 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_aegis_v6/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v6/distilbert/summary.csv | distilbert | aegis_v6 | 66955010 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 53449064 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 668262 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 1.2527x <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v6/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v6/gpt2/summary.csv | gpt2 | aegis_v6 | 124441344 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 97429452 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 1336524 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 1.2772x <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v6/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v6/roberta/summary.csv line 2 --> |
| results/unified_aegis_v6/roberta/summary.csv | roberta | aegis_v6 | 124647170 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 104722382 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 1336524 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 1.1903x <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_aegis_v6/distilbert/summary.csv | distilbert | full | 45229.11 tok/s <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v6/distilbert/summary.csv | distilbert | aegis_v6 | 11663.21 tok/s <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 301.26 MB <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v6/gpt2/summary.csv | gpt2 | full | 22868.32 tok/s <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v6/gpt2/summary.csv | gpt2 | aegis_v6 | 6170.91 tok/s <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 493.7 MB <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v6/roberta/summary.csv | roberta | full | 22893.69 tok/s <!-- source: results/unified_aegis_v6/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_aegis_v6/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 2 --> |
| results/unified_aegis_v6/roberta/summary.csv | roberta | aegis_v6 | 5763.15 tok/s <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 596.26 MB <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=93.30%, comp=1.252x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 93.30% | `results/rank_ablation/agnews_rank32/summary.csv` |
| F1 Score | 0.9290 | `results/rank_ablation/agnews_rank32/summary.csv` |
| Compression | 1.252x | `results/rank_ablation/agnews_rank32/summary.csv` |
| Compressed Layers | 0 | `results/rank_ablation/agnews_rank32/summary.csv` |
| Fallback Layers | 0 | `results/rank_ablation/agnews_rank32/summary.csv` |
| Timestamp | 2026-09-19T23:45:05.950130 | `results/rank_ablation/agnews_rank32/summary.csv` |

