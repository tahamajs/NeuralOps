# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_aegis_hybrid/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 2 --> |
| results/unified_aegis_hybrid/distilbert/summary.csv | distilbert | aegis_hybrid | PASS | 84.98% <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 0.8634 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> |
| results/unified_aegis_hybrid/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 2 --> |
| results/unified_aegis_hybrid/gpt2/summary.csv | gpt2 | aegis_hybrid | PASS | 59.52% <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 0.6735 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> |
| results/unified_aegis_hybrid/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 2 --> |
| results/unified_aegis_hybrid/roberta/summary.csv | roberta | aegis_hybrid | PASS_LOW | 51.26% <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 0.6698 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_aegis_hybrid/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 2 --> |
| results/unified_aegis_hybrid/distilbert/summary.csv | distilbert | aegis_hybrid | 66955010 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 59652626 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 2134800 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 1.1224x <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> |
| results/unified_aegis_hybrid/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 2 --> |
| results/unified_aegis_hybrid/gpt2/summary.csv | gpt2 | aegis_hybrid | 124441344 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 101579088 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 3090000 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 1.2251x <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> |
| results/unified_aegis_hybrid/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 2 --> |
| results/unified_aegis_hybrid/roberta/summary.csv | roberta | aegis_hybrid | 124647170 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 108872018 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 3090000 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 1.1449x <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_aegis_hybrid/distilbert/summary.csv | distilbert | full | 10631.82 tok/s <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 2 --> | 388.55 MB <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 2 --> |
| results/unified_aegis_hybrid/distilbert/summary.csv | distilbert | aegis_hybrid | 3802.96 tok/s <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 319.22 MB <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> |
| results/unified_aegis_hybrid/gpt2/summary.csv | gpt2 | full | 21011.47 tok/s <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 2 --> |
| results/unified_aegis_hybrid/gpt2/summary.csv | gpt2 | aegis_hybrid | 8474.53 tok/s <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 476.0 MB <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> |
| results/unified_aegis_hybrid/roberta/summary.csv | roberta | full | 23951.82 tok/s <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 2 --> |
| results/unified_aegis_hybrid/roberta/summary.csv | roberta | aegis_hybrid | 7060.79 tok/s <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 596.29 MB <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=93.00%, comp=1.050x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 93.00% | `results/rank_ablation/ablation_all_rows.csv` |
| F1 Score | 0.9269 | `results/rank_ablation/ablation_all_rows.csv` |
| Compression | 1.050x | `results/rank_ablation/ablation_all_rows.csv` |
| Compressed Layers | 0 | `results/rank_ablation/ablation_all_rows.csv` |
| Fallback Layers | 0 | `results/rank_ablation/ablation_all_rows.csv` |
| Timestamp | 2026-09-20T00:29:48.030674 | `results/rank_ablation/ablation_all_rows.csv` |

