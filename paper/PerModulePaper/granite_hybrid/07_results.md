# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_granite_hybrid/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 2 --> |
| results/unified_granite_hybrid/distilbert/summary.csv | distilbert | granite_hybrid | PASS | 63.65% <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> | 0.7325 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> |
| results/unified_granite_hybrid/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 2 --> |
| results/unified_granite_hybrid/gpt2/summary.csv | gpt2 | granite_hybrid | PASS_LOW | 51.72% <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> | 0.6774 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> |
<!-- FABRICATED: | results/unified_granite_hybrid/roberta/summary.csv | roberta | full | PASS | 75.0% <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 2 --> | 0.7578 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 2 --> | -->
| results/unified_granite_hybrid/roberta/summary.csv | roberta | granite_hybrid | PASS_LOW | 53.33% <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> | 0.6772 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_granite_hybrid/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 2 --> |
| results/unified_granite_hybrid/distilbert/summary.csv | distilbert | granite_hybrid | 66955010 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> | 216526142 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> | 1181984 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> | 0.3092x <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> |
| results/unified_granite_hybrid/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 2 --> |
| results/unified_granite_hybrid/gpt2/summary.csv | gpt2 | granite_hybrid | 124441344 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> | 423583608 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> | 1181256 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> | 0.2938x <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> |
| results/unified_granite_hybrid/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 2 --> |
| results/unified_granite_hybrid/roberta/summary.csv | roberta | granite_hybrid | 124647170 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> | 430876538 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> | 1771838 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> | 0.2893x <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_granite_hybrid/distilbert/summary.csv | distilbert | full | 44704.41 tok/s <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 2 --> |
| results/unified_granite_hybrid/distilbert/summary.csv | distilbert | granite_hybrid | 8212.62 tok/s <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> | 1604.75 MB <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> |
| results/unified_granite_hybrid/gpt2/summary.csv | gpt2 | full | 27504.15 tok/s <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 2 --> |
| results/unified_granite_hybrid/gpt2/summary.csv | gpt2 | granite_hybrid | 4214.39 tok/s <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> | 2393.89 MB <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> |
| results/unified_granite_hybrid/roberta/summary.csv | roberta | full | 26766.82 tok/s <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 2 --> |
| results/unified_granite_hybrid/roberta/summary.csv | roberta | granite_hybrid | 4243.84 tok/s <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> | 2501.8 MB <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.14% | `results/unified_sweep/distilbert/summary.csv` |
| F1 Score | 0.9040 | `results/unified_sweep/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary.csv` |
| Timestamp | 2026-09-18T23:45:39.547690 | `results/unified_sweep/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [57.91, 75.0]
> These numbers must NOT be cited in the final paper.

