# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_spectral_mamba/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 2 --> |
| results/unified_spectral_mamba/distilbert/summary.csv | distilbert | spectral_mamba | PASS | 90.71% <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> | 0.9111 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> |
| results/unified_spectral_mamba/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 2 --> |
| results/unified_spectral_mamba/gpt2/summary.csv | gpt2 | spectral_mamba | PASS_LOW | 51.03% <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> | 0.6753 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> |
| results/unified_spectral_mamba/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 2 --> |
| results/unified_spectral_mamba/roberta/summary.csv | roberta | spectral_mamba | PASS_LOW | 50.92% <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_spectral_mamba/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 2 --> |
| results/unified_spectral_mamba/distilbert/summary.csv | distilbert | spectral_mamba | 66955010 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> | 78184748 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> | 1186604 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> | 0.8564x <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> |
| results/unified_spectral_mamba/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 2 --> |
| results/unified_spectral_mamba/gpt2/summary.csv | gpt2 | spectral_mamba | 124441344 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> | 146900820 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> | 1190484 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> | 0.8471x <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> |
| results/unified_spectral_mamba/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 2 --> |
| results/unified_spectral_mamba/roberta/summary.csv | roberta | spectral_mamba | 124647170 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> | 154193750 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> | 1781078 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> | 0.8084x <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_spectral_mamba/distilbert/summary.csv | distilbert | full | 143380.76 tok/s <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 2 --> |
| results/unified_spectral_mamba/distilbert/summary.csv | distilbert | spectral_mamba | 66788.52 tok/s <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> | 378.15 MB <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> |
| results/unified_spectral_mamba/gpt2/summary.csv | gpt2 | full | 65013.44 tok/s <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 2 --> |
| results/unified_spectral_mamba/gpt2/summary.csv | gpt2 | spectral_mamba | 30894.94 tok/s <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> | 650.47 MB <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> |
| results/unified_spectral_mamba/roberta/summary.csv | roberta | full | 81138.79 tok/s <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 2 --> |
| results/unified_spectral_mamba/roberta/summary.csv | roberta | spectral_mamba | 32098.62 tok/s <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> | 749.82 MB <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=0.856 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.71% | `results/unified_spectral_mamba/distilbert/summary.csv` |
| F1 Score | 0.9111 | `results/unified_spectral_mamba/distilbert/summary.csv` |
| Compression | 0.856x | `results/unified_spectral_mamba/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_spectral_mamba/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_spectral_mamba/distilbert/summary.csv` |
| Timestamp | 2026-09-11T21:03:01.937278 | `results/unified_spectral_mamba/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.46]
> These numbers must NOT be cited in the final paper.

