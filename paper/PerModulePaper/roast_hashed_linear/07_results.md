# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_roast_hashed_linear/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 2 --> |
| results/unified_roast_hashed_linear/distilbert/summary.csv | distilbert | roast_hashed_linear | PASS | 91.06% <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 3 --> | 0.9137 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 3 --> |
| results/unified_roast_hashed_linear/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 2 --> |
| results/unified_roast_hashed_linear/gpt2/summary.csv | gpt2 | roast_hashed_linear | PASS_LOW | 51.72% <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 3 --> | 0.6774 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 3 --> |
| results/unified_roast_hashed_linear/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 2 --> |
| results/unified_roast_hashed_linear/roberta/summary.csv | roberta | roast_hashed_linear | PASS_LOW | 51.26% <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 3 --> | 0.6763 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_roast_hashed_linear/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 2 --> |
| results/unified_roast_hashed_linear/distilbert/summary.csv | distilbert | roast_hashed_linear | 66955010 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 3 --> | 6914 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 3 --> |
| results/unified_roast_hashed_linear/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 2 --> |
| results/unified_roast_hashed_linear/gpt2/summary.csv | gpt2 | roast_hashed_linear | 124441344 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 3 --> |
| results/unified_roast_hashed_linear/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 2 --> |
| results/unified_roast_hashed_linear/roberta/summary.csv | roberta | roast_hashed_linear | 124647170 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 3 --> | 11522 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_roast_hashed_linear/distilbert/summary.csv | distilbert | full | 138475.48 tok/s <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 2 --> |
| results/unified_roast_hashed_linear/distilbert/summary.csv | distilbert | roast_hashed_linear | 152016.37 tok/s <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 3 --> | 308.84 MB <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 3 --> | 1.0000503891044192 <!-- source: results/unified_roast_hashed_linear/distilbert/summary.csv line 3 --> |
| results/unified_roast_hashed_linear/gpt2/summary.csv | gpt2 | full | 65153.01 tok/s <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 2 --> |
| results/unified_roast_hashed_linear/gpt2/summary.csv | gpt2 | roast_hashed_linear | 66259.72 tok/s <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 3 --> | 0.9999008211824628 <!-- source: results/unified_roast_hashed_linear/gpt2/summary.csv line 3 --> |
| results/unified_roast_hashed_linear/roberta/summary.csv | roberta | full | 81101.95 tok/s <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 2 --> |
| results/unified_roast_hashed_linear/roberta/summary.csv | roberta | roast_hashed_linear | 81344.21 tok/s <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 3 --> | 528.86 MB <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 3 --> | 0.9963449521197212 <!-- source: results/unified_roast_hashed_linear/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 91.06% | `results/unified_roast_hashed_linear/distilbert/summary.csv` |
| F1 Score | 0.9137 | `results/unified_roast_hashed_linear/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_roast_hashed_linear/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_roast_hashed_linear/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_roast_hashed_linear/distilbert/summary.csv` |
| Timestamp | 2026-09-11T20:47:25.525519 | `results/unified_roast_hashed_linear/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [57.91, 75.46]
> These numbers must NOT be cited in the final paper.

