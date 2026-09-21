# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_hashed_linear/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 2 --> |
| results/unified_hashed_linear/distilbert/summary.csv | distilbert | hashed_linear | PASS | 89.68% <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 3 --> | 0.9009 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 3 --> |
| results/unified_hashed_linear/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 2 --> |
| results/unified_hashed_linear/gpt2/summary.csv | gpt2 | hashed_linear | PASS | 57.91% <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 3 --> | 0.6861 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 3 --> |
| results/unified_hashed_linear/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 2 --> |
| results/unified_hashed_linear/qwen3_5/summary.csv | qwen3_5 | hashed_linear | FAIL | 0.0% <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 3 --> | 0.557 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 3 --> |
| results/unified_hashed_linear/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_hashed_linear/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 2 --> |
| results/unified_hashed_linear/roberta/summary.csv | roberta | hashed_linear | PASS | 84.06% <!-- source: results/unified_hashed_linear/roberta/summary.csv line 3 --> | 0.8529 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_hashed_linear/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 2 --> |
| results/unified_hashed_linear/distilbert/summary.csv | distilbert | hashed_linear | 66955010 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 3 --> |
| results/unified_hashed_linear/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 2 --> |
| results/unified_hashed_linear/gpt2/summary.csv | gpt2 | hashed_linear | 124441344 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 3 --> |
| results/unified_hashed_linear/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 2 --> |
| results/unified_hashed_linear/qwen3_5/summary.csv | qwen3_5 | hashed_linear | 763217728 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 3 --> | 0 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 3 --> |
| results/unified_hashed_linear/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hashed_linear/roberta/summary.csv line 2 --> |
| results/unified_hashed_linear/roberta/summary.csv | roberta | hashed_linear | 124647170 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_hashed_linear/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_hashed_linear/distilbert/summary.csv | distilbert | full | 163034.79 tok/s <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 2 --> |
| results/unified_hashed_linear/distilbert/summary.csv | distilbert | hashed_linear | 172097.68 tok/s <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 3 --> | 339.34 MB <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hashed_linear/distilbert/summary.csv line 3 --> |
| results/unified_hashed_linear/gpt2/summary.csv | gpt2 | full | 71384.93 tok/s <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 2 --> |
| results/unified_hashed_linear/gpt2/summary.csv | gpt2 | hashed_linear | 71593.46 tok/s <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 3 --> | 637.04 MB <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hashed_linear/gpt2/summary.csv line 3 --> |
| results/unified_hashed_linear/qwen3_5/summary.csv | qwen3_5 | full | 1303.54 tok/s <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 2 --> |
| results/unified_hashed_linear/qwen3_5/summary.csv | qwen3_5 | hashed_linear | 1303.54 tok/s <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 3 --> | 1796.78 MB <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hashed_linear/qwen3_5/summary.csv line 3 --> |
| results/unified_hashed_linear/roberta/summary.csv | roberta | full | 112717.41 tok/s <!-- source: results/unified_hashed_linear/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_hashed_linear/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 2 --> |
| results/unified_hashed_linear/roberta/summary.csv | roberta | hashed_linear | 104843.18 tok/s <!-- source: results/unified_hashed_linear/roberta/summary.csv line 3 --> | 362.29 MB <!-- source: results/unified_hashed_linear/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hashed_linear/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 89.68% | `results/unified_hashed_linear/distilbert/summary.csv` |
| F1 Score | 0.9009 | `results/unified_hashed_linear/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_hashed_linear/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_hashed_linear/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_hashed_linear/distilbert/summary.csv` |
| Timestamp | 2026-09-10T21:07:56.031811 | `results/unified_hashed_linear/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [60.78, 67.89, 75.11]
> These numbers must NOT be cited in the final paper.

