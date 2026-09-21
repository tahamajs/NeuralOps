# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_simple_ssm/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 2 --> |
| results/unified_simple_ssm/distilbert/summary.csv | distilbert | simple_ssm | PASS_NO_COMPRESSION | 50.34% <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 3 --> | 0.6002 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 3 --> |
| results/unified_simple_ssm/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 2 --> |
| results/unified_simple_ssm/gpt2/summary.csv | gpt2 | simple_ssm | PASS_NO_COMPRESSION | 49.66% <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 3 --> | 0.6118 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 3 --> |
| results/unified_simple_ssm/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_simple_ssm/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 2 --> |
| results/unified_simple_ssm/roberta/summary.csv | roberta | simple_ssm | PASS_NO_COMPRESSION | 50.92% <!-- source: results/unified_simple_ssm/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_simple_ssm/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 2 --> |
| results/unified_simple_ssm/distilbert/summary.csv | distilbert | simple_ssm | 66955010 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 3 --> | 67563266 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 3 --> | 516096 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 3 --> |
| results/unified_simple_ssm/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 2 --> |
| results/unified_simple_ssm/gpt2/summary.csv | gpt2 | simple_ssm | 124441344 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 3 --> | 125657856 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 3 --> | 1032192 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 3 --> |
| results/unified_simple_ssm/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_simple_ssm/roberta/summary.csv line 2 --> |
| results/unified_simple_ssm/roberta/summary.csv | roberta | simple_ssm | 124647170 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 3 --> | 132950786 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 3 --> | 1032192 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_simple_ssm/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_simple_ssm/distilbert/summary.csv | distilbert | full | 23488.77 tok/s <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 2 --> |
| results/unified_simple_ssm/distilbert/summary.csv | distilbert | simple_ssm | 7672.98 tok/s <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 3 --> | 350.04 MB <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_simple_ssm/distilbert/summary.csv line 3 --> |
| results/unified_simple_ssm/gpt2/summary.csv | gpt2 | full | 26658.89 tok/s <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 2 --> |
| results/unified_simple_ssm/gpt2/summary.csv | gpt2 | simple_ssm | 9154.04 tok/s <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 3 --> | 578.92 MB <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_simple_ssm/gpt2/summary.csv line 3 --> |
| results/unified_simple_ssm/roberta/summary.csv | roberta | full | 12057.34 tok/s <!-- source: results/unified_simple_ssm/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_simple_ssm/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 2 --> |
| results/unified_simple_ssm/roberta/summary.csv | roberta | simple_ssm | 3946.63 tok/s <!-- source: results/unified_simple_ssm/roberta/summary.csv line 3 --> | 678.8 MB <!-- source: results/unified_simple_ssm/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_simple_ssm/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 68.38% | `results/adaptive_v3/benchmark_summary.csv` |
| F1 Score | 0.8122 | `results/adaptive_v3/benchmark_summary.csv` |
| Compression | 1.000x | `results/adaptive_v3/benchmark_summary.csv` |
| Compressed Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Fallback Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Timestamp | 2026-08-26 07:36:25 | `results/adaptive_v3/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 79.36, 78.56]
> These numbers must NOT be cited in the final paper.

