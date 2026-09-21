# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_lola_attention/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_lola_attention/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 2 --> |
| results/unified_lola_attention/distilbert/summary.csv | distilbert | lola_attention | PASS | 77.98% <!-- source: results/unified_lola_attention/distilbert/summary.csv line 3 --> | 0.7996 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 3 --> |
| results/unified_lola_attention/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_lola_attention/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 2 --> |
| results/unified_lola_attention/gpt2/summary.csv | gpt2 | lola_attention | PASS | 57.22% <!-- source: results/unified_lola_attention/gpt2/summary.csv line 3 --> | 0.4897 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 3 --> |
| results/unified_lola_attention/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_lola_attention/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_lola_attention/roberta/summary.csv line 2 --> |
| results/unified_lola_attention/roberta/summary.csv | roberta | lola_attention | PASS_LOW | 50.92% <!-- source: results/unified_lola_attention/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_lola_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_lola_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_lola_attention/distilbert/summary.csv line 2 --> |
| results/unified_lola_attention/distilbert/summary.csv | distilbert | lola_attention | 66955010 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 3 --> | 54568718 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 3 --> | 589836 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 3 --> | 1.227x <!-- source: results/unified_lola_attention/distilbert/summary.csv line 3 --> |
| results/unified_lola_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_lola_attention/gpt2/summary.csv line 2 --> |
| results/unified_lola_attention/gpt2/summary.csv | gpt2 | lola_attention | 124441344 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 3 --> | 99668760 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 3 --> | 1179672 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 3 --> | 1.2485x <!-- source: results/unified_lola_attention/gpt2/summary.csv line 3 --> |
| results/unified_lola_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_lola_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_lola_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_lola_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_lola_attention/roberta/summary.csv line 2 --> |
| results/unified_lola_attention/roberta/summary.csv | roberta | lola_attention | 124647170 <!-- source: results/unified_lola_attention/roberta/summary.csv line 3 --> | 106961690 <!-- source: results/unified_lola_attention/roberta/summary.csv line 3 --> | 1179672 <!-- source: results/unified_lola_attention/roberta/summary.csv line 3 --> | 1.1653x <!-- source: results/unified_lola_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_lola_attention/distilbert/summary.csv | distilbert | full | 59175.98 tok/s <!-- source: results/unified_lola_attention/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_lola_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 2 --> |
| results/unified_lola_attention/distilbert/summary.csv | distilbert | lola_attention | 18153.1 tok/s <!-- source: results/unified_lola_attention/distilbert/summary.csv line 3 --> | 260.06 MB <!-- source: results/unified_lola_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_lola_attention/distilbert/summary.csv line 3 --> |
| results/unified_lola_attention/gpt2/summary.csv | gpt2 | full | 26737.05 tok/s <!-- source: results/unified_lola_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_lola_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 2 --> |
| results/unified_lola_attention/gpt2/summary.csv | gpt2 | lola_attention | 11377.73 tok/s <!-- source: results/unified_lola_attention/gpt2/summary.csv line 3 --> | 467.79 MB <!-- source: results/unified_lola_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_lola_attention/gpt2/summary.csv line 3 --> |
| results/unified_lola_attention/roberta/summary.csv | roberta | full | 27468.41 tok/s <!-- source: results/unified_lola_attention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_lola_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_lola_attention/roberta/summary.csv line 2 --> |
| results/unified_lola_attention/roberta/summary.csv | roberta | lola_attention | 8481.68 tok/s <!-- source: results/unified_lola_attention/roberta/summary.csv line 3 --> | 808.59 MB <!-- source: results/unified_lola_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_lola_attention/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=77.98%, comp=1.227x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 77.98% | `results/unified_lola_attention/distilbert/summary.csv` |
| F1 Score | 0.7996 | `results/unified_lola_attention/distilbert/summary.csv` |
| Compression | 1.227x | `results/unified_lola_attention/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_lola_attention/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_lola_attention/distilbert/summary.csv` |
| Timestamp | 2026-09-15T04:21:57.496107 | `results/unified_lola_attention/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71]
> These numbers must NOT be cited in the final paper.

