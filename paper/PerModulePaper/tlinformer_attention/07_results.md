# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_tlinformer/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_tlinformer/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 2 --> |
| results/unified_tlinformer/distilbert/summary.csv | distilbert | tlinformer | PASS | 90.83% <!-- source: results/unified_tlinformer/distilbert/summary.csv line 3 --> | 0.9119 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 3 --> |
| results/unified_tlinformer/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_tlinformer/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 2 --> |
| results/unified_tlinformer/gpt2/summary.csv | gpt2 | tlinformer | PASS_LOW | 51.03% <!-- source: results/unified_tlinformer/gpt2/summary.csv line 3 --> | 0.6753 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 3 --> |
| results/unified_tlinformer/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_tlinformer/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_tlinformer/roberta/summary.csv line 2 --> |
| results/unified_tlinformer/roberta/summary.csv | roberta | tlinformer | PASS_LOW | 50.92% <!-- source: results/unified_tlinformer/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_tlinformer/roberta/summary.csv line 3 --> |
| results/unified_tlinformer_attention/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 2 --> |
| results/unified_tlinformer_attention/distilbert/summary.csv | distilbert | tlinformer_attention | PASS | 90.25% <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 3 --> | 0.9063 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 3 --> |
| results/unified_tlinformer_attention/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 2 --> |
| results/unified_tlinformer_attention/gpt2/summary.csv | gpt2 | tlinformer_attention | PASS_LOW | 51.03% <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 3 --> | 0.6753 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 3 --> |
| results/unified_tlinformer_attention/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 2 --> |
| results/unified_tlinformer_attention/roberta/summary.csv | roberta | tlinformer_attention | PASS_LOW | 53.56% <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 3 --> | 0.6853 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_tlinformer/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tlinformer/distilbert/summary.csv line 2 --> |
| results/unified_tlinformer/distilbert/summary.csv | distilbert | tlinformer | 66955010 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 3 --> | 67938080 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 3 --> | 1575200 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 3 --> | 0.9855x <!-- source: results/unified_tlinformer/distilbert/summary.csv line 3 --> |
| results/unified_tlinformer/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tlinformer/gpt2/summary.csv line 2 --> |
| results/unified_tlinformer/gpt2/summary.csv | gpt2 | tlinformer | 124441344 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 3 --> | 127193916 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 3 --> | 1181244 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 3 --> | 0.9784x <!-- source: results/unified_tlinformer/gpt2/summary.csv line 3 --> |
| results/unified_tlinformer/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_tlinformer/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_tlinformer/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_tlinformer/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tlinformer/roberta/summary.csv line 2 --> |
| results/unified_tlinformer/roberta/summary.csv | roberta | tlinformer | 124647170 <!-- source: results/unified_tlinformer/roberta/summary.csv line 3 --> | 133703486 <!-- source: results/unified_tlinformer/roberta/summary.csv line 3 --> | 1182014 <!-- source: results/unified_tlinformer/roberta/summary.csv line 3 --> | 0.9323x <!-- source: results/unified_tlinformer/roberta/summary.csv line 3 --> |
| results/unified_tlinformer_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 2 --> |
| results/unified_tlinformer_attention/distilbert/summary.csv | distilbert | tlinformer_attention | 66955010 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 3 --> | 67741472 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 3 --> | 1378592 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 3 --> | 0.9884x <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 3 --> |
| results/unified_tlinformer_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 2 --> |
| results/unified_tlinformer_attention/gpt2/summary.csv | gpt2 | tlinformer_attention | 124441344 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 3 --> | 126407484 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 3 --> | 1967676 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 3 --> | 0.9844x <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 3 --> |
| results/unified_tlinformer_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 2 --> |
| results/unified_tlinformer_attention/roberta/summary.csv | roberta | tlinformer_attention | 124647170 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 3 --> | 133308734 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 3 --> | 1576766 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 3 --> | 0.935x <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_tlinformer/distilbert/summary.csv | distilbert | full | 139742.52 tok/s <!-- source: results/unified_tlinformer/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_tlinformer/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 2 --> |
| results/unified_tlinformer/distilbert/summary.csv | distilbert | tlinformer | 70831.26 tok/s <!-- source: results/unified_tlinformer/distilbert/summary.csv line 3 --> | 335.97 MB <!-- source: results/unified_tlinformer/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tlinformer/distilbert/summary.csv line 3 --> |
| results/unified_tlinformer/gpt2/summary.csv | gpt2 | full | 61495.11 tok/s <!-- source: results/unified_tlinformer/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_tlinformer/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 2 --> |
| results/unified_tlinformer/gpt2/summary.csv | gpt2 | tlinformer | 35240.24 tok/s <!-- source: results/unified_tlinformer/gpt2/summary.csv line 3 --> | 574.72 MB <!-- source: results/unified_tlinformer/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tlinformer/gpt2/summary.csv line 3 --> |
| results/unified_tlinformer/roberta/summary.csv | roberta | full | 79049.94 tok/s <!-- source: results/unified_tlinformer/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_tlinformer/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tlinformer/roberta/summary.csv line 2 --> |
| results/unified_tlinformer/roberta/summary.csv | roberta | tlinformer | 34963.34 tok/s <!-- source: results/unified_tlinformer/roberta/summary.csv line 3 --> | 664.64 MB <!-- source: results/unified_tlinformer/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tlinformer/roberta/summary.csv line 3 --> |
| results/unified_tlinformer_attention/distilbert/summary.csv | distilbert | full | 137757.17 tok/s <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 2 --> |
| results/unified_tlinformer_attention/distilbert/summary.csv | distilbert | tlinformer_attention | 75937.59 tok/s <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 3 --> | 331.47 MB <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tlinformer_attention/distilbert/summary.csv line 3 --> |
| results/unified_tlinformer_attention/gpt2/summary.csv | gpt2 | full | 63407.14 tok/s <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 2 --> |
| results/unified_tlinformer_attention/gpt2/summary.csv | gpt2 | tlinformer_attention | 35641.54 tok/s <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 3 --> | 571.59 MB <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tlinformer_attention/gpt2/summary.csv line 3 --> |
| results/unified_tlinformer_attention/roberta/summary.csv | roberta | full | 82711.85 tok/s <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 2 --> |
| results/unified_tlinformer_attention/roberta/summary.csv | roberta | tlinformer_attention | 36981.95 tok/s <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 3 --> | 660.78 MB <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tlinformer_attention/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=0.988 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.25% | `results/unified_tlinformer_attention/distilbert/summary.csv` |
| F1 Score | 0.9063 | `results/unified_tlinformer_attention/distilbert/summary.csv` |
| Compression | 0.988x | `results/unified_tlinformer_attention/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_tlinformer_attention/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_tlinformer_attention/distilbert/summary.csv` |
| Timestamp | 2026-09-11T21:52:44.485272 | `results/unified_tlinformer_attention/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [57.91, 75.46, 57.91, 75.46]
> These numbers must NOT be cited in the final paper.

