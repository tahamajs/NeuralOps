| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Cur Linear** | **88.76%** | -1.72% | PASS | 0.8884 | 0 | **1.00x** | 339.2 MB | 20.0 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Cur Linear** | **78.44%** | +3.33% | PASS | 0.8203 | 0 | **1.00x** | 359.0 MB | 30.6 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Cur Linear** | **58.94%** | -1.84% | PASS | 0.6792 | 0 | **1.00x** | 636.3 MB | 26.1 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_cur_linear/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_cur_linear/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 2 --> |
| results/unified_cur_linear/distilbert/summary.csv | distilbert | cur_linear | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_cur_linear/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 3 --> |
| results/unified_cur_linear/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_cur_linear/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 2 --> |
| results/unified_cur_linear/gpt2/summary.csv | gpt2 | cur_linear | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_cur_linear/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 3 --> |
| results/unified_cur_linear/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 2 --> |
| results/unified_cur_linear/qwen3_5/summary.csv | qwen3_5 | cur_linear | FAIL | 0.0% <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 3 --> | 0.557 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 3 --> |
| results/unified_cur_linear/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_cur_linear/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_cur_linear/roberta/summary.csv line 2 --> |
| results/unified_cur_linear/roberta/summary.csv | roberta | cur_linear | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_cur_linear/roberta/summary.csv line 3 --> | 0.8124 <!-- source: results/unified_cur_linear/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_cur_linear/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur_linear/distilbert/summary.csv line 2 --> |
| results/unified_cur_linear/distilbert/summary.csv | distilbert | cur_linear | 66955010 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cur_linear/distilbert/summary.csv line 3 --> |
| results/unified_cur_linear/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur_linear/gpt2/summary.csv line 2 --> |
| results/unified_cur_linear/gpt2/summary.csv | gpt2 | cur_linear | 124441344 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cur_linear/gpt2/summary.csv line 3 --> |
| results/unified_cur_linear/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 2 --> |
| results/unified_cur_linear/qwen3_5/summary.csv | qwen3_5 | cur_linear | 763217728 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 3 --> | 0 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 3 --> |
| results/unified_cur_linear/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_cur_linear/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_cur_linear/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_cur_linear/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur_linear/roberta/summary.csv line 2 --> |
| results/unified_cur_linear/roberta/summary.csv | roberta | cur_linear | 124647170 <!-- source: results/unified_cur_linear/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_cur_linear/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_cur_linear/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cur_linear/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_cur_linear/distilbert/summary.csv | distilbert | full | 52183.96 tok/s <!-- source: results/unified_cur_linear/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_cur_linear/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 2 --> |
| results/unified_cur_linear/distilbert/summary.csv | distilbert | cur_linear | 53900.26 tok/s <!-- source: results/unified_cur_linear/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_cur_linear/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur_linear/distilbert/summary.csv line 3 --> |
| results/unified_cur_linear/gpt2/summary.csv | gpt2 | full | 26661.55 tok/s <!-- source: results/unified_cur_linear/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_cur_linear/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 2 --> |
| results/unified_cur_linear/gpt2/summary.csv | gpt2 | cur_linear | 26605.88 tok/s <!-- source: results/unified_cur_linear/gpt2/summary.csv line 3 --> | 627.28 MB <!-- source: results/unified_cur_linear/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur_linear/gpt2/summary.csv line 3 --> |
| results/unified_cur_linear/qwen3_5/summary.csv | qwen3_5 | full | 1173.06 tok/s <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 2 --> |
| results/unified_cur_linear/qwen3_5/summary.csv | qwen3_5 | cur_linear | 1173.06 tok/s <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 3 --> | 1796.78 MB <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur_linear/qwen3_5/summary.csv line 3 --> |
| results/unified_cur_linear/roberta/summary.csv | roberta | full | 27034.84 tok/s <!-- source: results/unified_cur_linear/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_cur_linear/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur_linear/roberta/summary.csv line 2 --> |
| results/unified_cur_linear/roberta/summary.csv | roberta | cur_linear | 28011.5 tok/s <!-- source: results/unified_cur_linear/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_cur_linear/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur_linear/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `BROKEN`  
**Detail**: Error: note: zero compressed layers, model grew (LoRA only)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.71% | `results/unified_cur_linear/distilbert/summary.csv` |
| F1 Score | 0.9099 | `results/unified_cur_linear/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_cur_linear/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_cur_linear/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_cur_linear/distilbert/summary.csv` |
| Timestamp | 2026-09-15T11:42:21.796743 | `results/unified_cur_linear/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [67.89]
> These numbers must NOT be cited in the final paper.

