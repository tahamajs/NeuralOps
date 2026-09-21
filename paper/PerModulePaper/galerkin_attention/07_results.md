# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_galerkin/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_galerkin/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_galerkin/distilbert/summary.csv line 2 --> |
| results/unified_galerkin/distilbert/summary.csv | distilbert | galerkin | PASS_NO_COMPRESSION | 65.94% <!-- source: results/unified_galerkin/distilbert/summary.csv line 3 --> | 0.6426 <!-- source: results/unified_galerkin/distilbert/summary.csv line 3 --> |
| results/unified_galerkin/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_galerkin/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_galerkin/gpt2/summary.csv line 2 --> |
| results/unified_galerkin/gpt2/summary.csv | gpt2 | galerkin | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_galerkin/gpt2/summary.csv line 3 --> | 0.7982 <!-- source: results/unified_galerkin/gpt2/summary.csv line 3 --> |
| results/unified_galerkin/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_galerkin/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_galerkin/roberta/summary.csv line 2 --> |
| results/unified_galerkin/roberta/summary.csv | roberta | galerkin | PASS_NO_COMPRESSION | 49.2% <!-- source: results/unified_galerkin/roberta/summary.csv line 3 --> | 0.4656 <!-- source: results/unified_galerkin/roberta/summary.csv line 3 --> |
| results/unified_galerkin_attention/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 2 --> |
| results/unified_galerkin_attention/distilbert/summary.csv | distilbert | galerkin_attention | PASS_NO_COMPRESSION | 65.94% <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> | 0.6426 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> |
| results/unified_galerkin_attention/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 2 --> |
| results/unified_galerkin_attention/gpt2/summary.csv | gpt2 | galerkin_attention | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> | 0.7982 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> |
| results/unified_galerkin_attention/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 2 --> |
| results/unified_galerkin_attention/roberta/summary.csv | roberta | galerkin_attention | PASS_NO_COMPRESSION | 49.2% <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> | 0.4656 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_galerkin/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_galerkin/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_galerkin/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_galerkin/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_galerkin/distilbert/summary.csv line 2 --> |
| results/unified_galerkin/distilbert/summary.csv | distilbert | galerkin | 66955010 <!-- source: results/unified_galerkin/distilbert/summary.csv line 3 --> | 67544834 <!-- source: results/unified_galerkin/distilbert/summary.csv line 3 --> | 589824 <!-- source: results/unified_galerkin/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_galerkin/distilbert/summary.csv line 3 --> |
| results/unified_galerkin/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_galerkin/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_galerkin/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_galerkin/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_galerkin/gpt2/summary.csv line 2 --> |
| results/unified_galerkin/gpt2/summary.csv | gpt2 | galerkin | 124441344 <!-- source: results/unified_galerkin/gpt2/summary.csv line 3 --> | 125620992 <!-- source: results/unified_galerkin/gpt2/summary.csv line 3 --> | 1179648 <!-- source: results/unified_galerkin/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_galerkin/gpt2/summary.csv line 3 --> |
| results/unified_galerkin/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_galerkin/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_galerkin/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_galerkin/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_galerkin/roberta/summary.csv line 2 --> |
| results/unified_galerkin/roberta/summary.csv | roberta | galerkin | 124647170 <!-- source: results/unified_galerkin/roberta/summary.csv line 3 --> | 132913922 <!-- source: results/unified_galerkin/roberta/summary.csv line 3 --> | 1179648 <!-- source: results/unified_galerkin/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_galerkin/roberta/summary.csv line 3 --> |
| results/unified_galerkin_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 2 --> |
| results/unified_galerkin_attention/distilbert/summary.csv | distilbert | galerkin_attention | 66955010 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> | 67544834 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> | 589824 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> |
| results/unified_galerkin_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 2 --> |
| results/unified_galerkin_attention/gpt2/summary.csv | gpt2 | galerkin_attention | 124441344 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> | 125620992 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> | 1179648 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> |
| results/unified_galerkin_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 2 --> |
| results/unified_galerkin_attention/roberta/summary.csv | roberta | galerkin_attention | 124647170 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> | 132913922 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> | 1179648 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_galerkin/distilbert/summary.csv | distilbert | full | 53548.04 tok/s <!-- source: results/unified_galerkin/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_galerkin/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_galerkin/distilbert/summary.csv line 2 --> |
| results/unified_galerkin/distilbert/summary.csv | distilbert | galerkin | 16310.58 tok/s <!-- source: results/unified_galerkin/distilbert/summary.csv line 3 --> | 322.36 MB <!-- source: results/unified_galerkin/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_galerkin/distilbert/summary.csv line 3 --> |
| results/unified_galerkin/gpt2/summary.csv | gpt2 | full | 28080.02 tok/s <!-- source: results/unified_galerkin/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_galerkin/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_galerkin/gpt2/summary.csv line 2 --> |
| results/unified_galerkin/gpt2/summary.csv | gpt2 | galerkin | 12433.25 tok/s <!-- source: results/unified_galerkin/gpt2/summary.csv line 3 --> | 567.91 MB <!-- source: results/unified_galerkin/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_galerkin/gpt2/summary.csv line 3 --> |
| results/unified_galerkin/roberta/summary.csv | roberta | full | 28429.45 tok/s <!-- source: results/unified_galerkin/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_galerkin/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_galerkin/roberta/summary.csv line 2 --> |
| results/unified_galerkin/roberta/summary.csv | roberta | galerkin | 8148.5 tok/s <!-- source: results/unified_galerkin/roberta/summary.csv line 3 --> | 653.87 MB <!-- source: results/unified_galerkin/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_galerkin/roberta/summary.csv line 3 --> |
| results/unified_galerkin_attention/distilbert/summary.csv | distilbert | full | 47421.57 tok/s <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 2 --> |
| results/unified_galerkin_attention/distilbert/summary.csv | distilbert | galerkin_attention | 15725.49 tok/s <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> | 322.36 MB <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> |
| results/unified_galerkin_attention/gpt2/summary.csv | gpt2 | full | 26733.0 tok/s <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 2 --> |
| results/unified_galerkin_attention/gpt2/summary.csv | gpt2 | galerkin_attention | 12149.25 tok/s <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> | 567.91 MB <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> |
| results/unified_galerkin_attention/roberta/summary.csv | roberta | full | 27040.11 tok/s <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 2 --> |
| results/unified_galerkin_attention/roberta/summary.csv | roberta | galerkin_attention | 7834.15 tok/s <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> | 653.87 MB <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 68.63% | `results/adaptive_v3/benchmark_summary.csv` |
| F1 Score | 0.8101 | `results/adaptive_v3/benchmark_summary.csv` |
| Compression | 1.000x | `results/adaptive_v3/benchmark_summary.csv` |
| Compressed Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Fallback Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Timestamp | 2026-08-26 08:18:34 | `results/adaptive_v3/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 90.71]
> These numbers must NOT be cited in the final paper.

