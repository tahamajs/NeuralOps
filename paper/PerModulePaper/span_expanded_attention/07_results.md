# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_span_expanded/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_span_expanded/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 2 --> |
| results/unified_span_expanded/distilbert/summary.csv | distilbert | span_expanded | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_span_expanded/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 3 --> |
| results/unified_span_expanded/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_span_expanded/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 2 --> |
| results/unified_span_expanded/gpt2/summary.csv | gpt2 | span_expanded | PASS_NO_COMPRESSION | 79.24% <!-- source: results/unified_span_expanded/gpt2/summary.csv line 3 --> | 0.7964 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 3 --> |
| results/unified_span_expanded/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_span_expanded/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_span_expanded/roberta/summary.csv line 2 --> |
| results/unified_span_expanded/roberta/summary.csv | roberta | span_expanded | PASS_NO_COMPRESSION | 81.65% <!-- source: results/unified_span_expanded/roberta/summary.csv line 3 --> | 0.8268 <!-- source: results/unified_span_expanded/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_span_expanded/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_span_expanded/distilbert/summary.csv line 2 --> |
| results/unified_span_expanded/distilbert/summary.csv | distilbert | span_expanded | 66955010 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 3 --> | 67544858 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 3 --> | 589848 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_span_expanded/distilbert/summary.csv line 3 --> |
| results/unified_span_expanded/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_span_expanded/gpt2/summary.csv line 2 --> |
| results/unified_span_expanded/gpt2/summary.csv | gpt2 | span_expanded | 124441344 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 3 --> | 125621040 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 3 --> | 1179696 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_span_expanded/gpt2/summary.csv line 3 --> |
| results/unified_span_expanded/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_span_expanded/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_span_expanded/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_span_expanded/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_span_expanded/roberta/summary.csv line 2 --> |
| results/unified_span_expanded/roberta/summary.csv | roberta | span_expanded | 124647170 <!-- source: results/unified_span_expanded/roberta/summary.csv line 3 --> | 132913970 <!-- source: results/unified_span_expanded/roberta/summary.csv line 3 --> | 1179696 <!-- source: results/unified_span_expanded/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_span_expanded/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_span_expanded/distilbert/summary.csv | distilbert | full | 27878.17 tok/s <!-- source: results/unified_span_expanded/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_span_expanded/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 2 --> |
| results/unified_span_expanded/distilbert/summary.csv | distilbert | span_expanded | 12805.16 tok/s <!-- source: results/unified_span_expanded/distilbert/summary.csv line 3 --> | 326.45 MB <!-- source: results/unified_span_expanded/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_span_expanded/distilbert/summary.csv line 3 --> |
| results/unified_span_expanded/gpt2/summary.csv | gpt2 | full | 14774.8 tok/s <!-- source: results/unified_span_expanded/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_span_expanded/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 2 --> |
| results/unified_span_expanded/gpt2/summary.csv | gpt2 | span_expanded | 6598.64 tok/s <!-- source: results/unified_span_expanded/gpt2/summary.csv line 3 --> | 568.95 MB <!-- source: results/unified_span_expanded/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_span_expanded/gpt2/summary.csv line 3 --> |
| results/unified_span_expanded/roberta/summary.csv | roberta | full | 14453.13 tok/s <!-- source: results/unified_span_expanded/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_span_expanded/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_span_expanded/roberta/summary.csv line 2 --> |
| results/unified_span_expanded/roberta/summary.csv | roberta | span_expanded | 6547.07 tok/s <!-- source: results/unified_span_expanded/roberta/summary.csv line 3 --> | 657.23 MB <!-- source: results/unified_span_expanded/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_span_expanded/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `NO_DATA`  
**Detail**: No CSV rows found

*No valid CSV data found for this module.*


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 90.71, 79.36, 79.24, 78.56, 81.65]
> These numbers must NOT be cited in the final paper.

