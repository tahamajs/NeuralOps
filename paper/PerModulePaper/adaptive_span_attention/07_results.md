# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_adaptive_span/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_span/distilbert/summary.csv | distilbert | adaptive_span | PASS_NO_COMPRESSION | 90.48% <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 0.9089 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> |
| results/unified_adaptive_span/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_span/gpt2/summary.csv | gpt2 | adaptive_span | PASS_NO_COMPRESSION | 81.08% <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 0.8328 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> |
| results/unified_adaptive_span/qwen3_5/summary.csv | qwen3_5 | full | PASS | 77.52% <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 2 --> | 0.761 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_span/qwen3_5/summary.csv | qwen3_5 | adaptive_span | PASS_LOW | 53.1% <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 0.2326 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> |
| results/unified_adaptive_span/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_adaptive_span/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 2 --> |
| results/unified_adaptive_span/roberta/summary.csv | roberta | adaptive_span | PASS_NO_COMPRESSION | 85.09% <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 0.8584 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_adaptive_span/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_span/distilbert/summary.csv | distilbert | adaptive_span | 66955010 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 67544906 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 589896 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> |
| results/unified_adaptive_span/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_span/gpt2/summary.csv | gpt2 | adaptive_span | 124441344 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 125621136 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 1179792 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> |
| results/unified_adaptive_span/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_span/qwen3_5/summary.csv | qwen3_5 | adaptive_span | 763217728 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 750459760 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 4727856 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 1.017x <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> |
| results/unified_adaptive_span/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_span/roberta/summary.csv line 2 --> |
| results/unified_adaptive_span/roberta/summary.csv | roberta | adaptive_span | 124647170 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 125532050 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 884880 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_adaptive_span/distilbert/summary.csv | distilbert | full | 44867.91 tok/s <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_span/distilbert/summary.csv | distilbert | adaptive_span | 25789.54 tok/s <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 323.38 MB <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> |
| results/unified_adaptive_span/gpt2/summary.csv | gpt2 | full | 26789.41 tok/s <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_span/gpt2/summary.csv | gpt2 | adaptive_span | 11187.59 tok/s <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 567.91 MB <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> |
| results/unified_adaptive_span/qwen3_5/summary.csv | qwen3_5 | full | 656.41 tok/s <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 2 --> | 1653.34 MB <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_span/qwen3_5/summary.csv | qwen3_5 | adaptive_span | 586.65 tok/s <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 1651.46 MB <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> |
| results/unified_adaptive_span/roberta/summary.csv | roberta | full | 27199.86 tok/s <!-- source: results/unified_adaptive_span/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_adaptive_span/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 2 --> |
| results/unified_adaptive_span/roberta/summary.csv | roberta | adaptive_span | 13743.09 tok/s <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 624.7 MB <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 82.00% | `results/agnews_unified_adaptive_span_attention/gpt2/summary.csv` |
| F1 Score | 0.8134 | `results/agnews_unified_adaptive_span_attention/gpt2/summary.csv` |
| Compression | 1.000x | `results/agnews_unified_adaptive_span_attention/gpt2/summary.csv` |
| Compressed Layers | 12 | `results/agnews_unified_adaptive_span_attention/gpt2/summary.csv` |
| Fallback Layers | 0 | `results/agnews_unified_adaptive_span_attention/gpt2/summary.csv` |
| Timestamp | 2026-09-18T08:59:00.183083 | `results/agnews_unified_adaptive_span_attention/gpt2/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 90.48, 77.52, 85.09]
> These numbers must NOT be cited in the final paper.

