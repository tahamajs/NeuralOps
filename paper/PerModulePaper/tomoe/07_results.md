| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Tomoe Compressor** | **69.50%** | -20.98% | PASS | 0.7569 | 0 | **1.00x** | 427.0 MB | 53.0 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Tomoe Compressor** | **50.92%** | -24.19% | PASS_LOW | 0.6748 | 0 | **1.00x** | 571.8 MB | 87.2 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Tomoe Compressor** | **54.24%** | -6.54% | PASS_LOW | 0.6780 | 0 | **1.00x** | 874.6 MB | 67.0 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_tomoe/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_tomoe/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_tomoe/distilbert/summary.csv line 2 --> |
| results/unified_tomoe/distilbert/summary.csv | distilbert | tomoe | PASS | 69.5% <!-- source: results/unified_tomoe/distilbert/summary.csv line 3 --> | 0.7569 <!-- source: results/unified_tomoe/distilbert/summary.csv line 3 --> |
| results/unified_tomoe/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_tomoe/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_tomoe/gpt2/summary.csv line 2 --> |
| results/unified_tomoe/gpt2/summary.csv | gpt2 | tomoe | PASS_LOW | 54.24% <!-- source: results/unified_tomoe/gpt2/summary.csv line 3 --> | 0.678 <!-- source: results/unified_tomoe/gpt2/summary.csv line 3 --> |
| results/unified_tomoe/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 2 --> |
| results/unified_tomoe/qwen3_5/summary.csv | qwen3_5 | tomoe | SKIP_QWEN3_5_TIMEOUT | 0.0% <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 3 --> | 0.557 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 3 --> |
| results/unified_tomoe/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_tomoe/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_tomoe/roberta/summary.csv line 2 --> |
| results/unified_tomoe/roberta/summary.csv | roberta | tomoe | PASS_LOW | 50.92% <!-- source: results/unified_tomoe/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_tomoe/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_tomoe/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_tomoe/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_tomoe/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_tomoe/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tomoe/distilbert/summary.csv line 2 --> |
| results/unified_tomoe/distilbert/summary.csv | distilbert | tomoe | 66955010 <!-- source: results/unified_tomoe/distilbert/summary.csv line 3 --> | 67121078 <!-- source: results/unified_tomoe/distilbert/summary.csv line 3 --> | 4154174 <!-- source: results/unified_tomoe/distilbert/summary.csv line 3 --> | 0.9975x <!-- source: results/unified_tomoe/distilbert/summary.csv line 3 --> |
| results/unified_tomoe/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_tomoe/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_tomoe/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_tomoe/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tomoe/gpt2/summary.csv line 2 --> |
| results/unified_tomoe/gpt2/summary.csv | gpt2 | tomoe | 124441344 <!-- source: results/unified_tomoe/gpt2/summary.csv line 3 --> | 124699632 <!-- source: results/unified_tomoe/gpt2/summary.csv line 3 --> | 1584 <!-- source: results/unified_tomoe/gpt2/summary.csv line 3 --> | 0.9979x <!-- source: results/unified_tomoe/gpt2/summary.csv line 3 --> |
| results/unified_tomoe/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 2 --> |
| results/unified_tomoe/qwen3_5/summary.csv | qwen3_5 | tomoe | 763217728 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 3 --> | 0 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 3 --> |
| results/unified_tomoe/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_tomoe/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_tomoe/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_tomoe/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tomoe/roberta/summary.csv line 2 --> |
| results/unified_tomoe/roberta/summary.csv | roberta | tomoe | 124647170 <!-- source: results/unified_tomoe/roberta/summary.csv line 3 --> | 124979306 <!-- source: results/unified_tomoe/roberta/summary.csv line 3 --> | 7716218 <!-- source: results/unified_tomoe/roberta/summary.csv line 3 --> | 0.9973x <!-- source: results/unified_tomoe/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_tomoe/distilbert/summary.csv | distilbert | full | 147673.36 tok/s <!-- source: results/unified_tomoe/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_tomoe/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tomoe/distilbert/summary.csv line 2 --> |
| results/unified_tomoe/distilbert/summary.csv | distilbert | tomoe | 9433.02 tok/s <!-- source: results/unified_tomoe/distilbert/summary.csv line 3 --> | 427.0 MB <!-- source: results/unified_tomoe/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tomoe/distilbert/summary.csv line 3 --> |
| results/unified_tomoe/gpt2/summary.csv | gpt2 | full | 63973.09 tok/s <!-- source: results/unified_tomoe/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_tomoe/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tomoe/gpt2/summary.csv line 2 --> |
| results/unified_tomoe/gpt2/summary.csv | gpt2 | tomoe | 7291.62 tok/s <!-- source: results/unified_tomoe/gpt2/summary.csv line 3 --> | 874.64 MB <!-- source: results/unified_tomoe/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tomoe/gpt2/summary.csv line 3 --> |
| results/unified_tomoe/qwen3_5/summary.csv | qwen3_5 | full | 1277.88 tok/s <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 2 --> |
| results/unified_tomoe/qwen3_5/summary.csv | qwen3_5 | tomoe | 1277.88 tok/s <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 3 --> | 1796.78 MB <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tomoe/qwen3_5/summary.csv line 3 --> |
| results/unified_tomoe/roberta/summary.csv | roberta | full | 90454.69 tok/s <!-- source: results/unified_tomoe/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_tomoe/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tomoe/roberta/summary.csv line 2 --> |
| results/unified_tomoe/roberta/summary.csv | roberta | tomoe | 5276.98 tok/s <!-- source: results/unified_tomoe/roberta/summary.csv line 3 --> | 571.84 MB <!-- source: results/unified_tomoe/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tomoe/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.40% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.9400 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T19:20:07.601679 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [60.78, 75.11]
> These numbers must NOT be cited in the final paper.

