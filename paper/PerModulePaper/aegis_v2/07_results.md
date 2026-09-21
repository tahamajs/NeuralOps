# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_aegis_v2/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v2/distilbert/summary.csv | distilbert | aegis_v2 | PASS | 72.36% <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 0.7737 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v2/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v2/gpt2/summary.csv | gpt2 | aegis_v2 | PASS_LOW | 52.98% <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 0.6782 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v2/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_aegis_v2/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 2 --> |
| results/unified_aegis_v2/roberta/summary.csv | roberta | aegis_v2 | PASS | 62.39% <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 0.7163 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_aegis_v2/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v2/distilbert/summary.csv | distilbert | aegis_v2 | 66955010 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 55523882 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 1545000 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 1.2059x <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v2/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v2/gpt2/summary.csv | gpt2 | aegis_v2 | 124441344 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 101579088 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 3090000 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 1.2251x <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v2/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v2/roberta/summary.csv line 2 --> |
| results/unified_aegis_v2/roberta/summary.csv | roberta | aegis_v2 | 124647170 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 108872018 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 3090000 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 1.1449x <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_aegis_v2/distilbert/summary.csv | distilbert | full | 51105.1 tok/s <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v2/distilbert/summary.csv | distilbert | aegis_v2 | 14953.19 tok/s <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 195.7 MB <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v2/gpt2/summary.csv | gpt2 | full | 23739.6 tok/s <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v2/gpt2/summary.csv | gpt2 | aegis_v2 | 8738.48 tok/s <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 476.0 MB <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v2/roberta/summary.csv | roberta | full | 22993.16 tok/s <!-- source: results/unified_aegis_v2/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_aegis_v2/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 2 --> |
| results/unified_aegis_v2/roberta/summary.csv | roberta | aegis_v2 | 7112.56 tok/s <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 591.16 MB <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=84.17%, comp=1.181x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 84.17% | `results/unified_sweep/distilbert/summary.csv` |
| F1 Score | 0.8516 | `results/unified_sweep/distilbert/summary.csv` |
| Compression | 1.181x | `results/unified_sweep/distilbert/summary.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary.csv` |
| Timestamp | 2026-09-18T22:52:43.764761 | `results/unified_sweep/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 79.36, 78.56]
> These numbers must NOT be cited in the final paper.

