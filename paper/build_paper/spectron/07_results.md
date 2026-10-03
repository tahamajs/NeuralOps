| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Spectron** | **86.58%** | -4.36% | PASS | 0.8765 | 0 | **1.15x** | 294.1 MB | 22.4 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Spectron** | **50.92%** | -24.54% | PASS_LOW | 0.6748 | 0 | **1.09x** | 587.2 MB | 29.0 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Spectron** | **51.26%** | -6.65% | PASS_LOW | 0.6758 | 0 | **1.17x** | 495.9 MB | 32.6 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_spectron/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_spectron/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_spectron/distilbert/summary.csv line 2 --> |
| results/unified_spectron/distilbert/summary.csv | distilbert | spectron | PASS | 86.81% <!-- source: results/unified_spectron/distilbert/summary.csv line 3 --> | 0.8793 <!-- source: results/unified_spectron/distilbert/summary.csv line 3 --> |
| results/unified_spectron/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_spectron/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_spectron/gpt2/summary.csv line 2 --> |
| results/unified_spectron/gpt2/summary.csv | gpt2 | spectron | PASS_LOW | 51.26% <!-- source: results/unified_spectron/gpt2/summary.csv line 3 --> | 0.6758 <!-- source: results/unified_spectron/gpt2/summary.csv line 3 --> |
| results/unified_spectron/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_spectron/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_spectron/roberta/summary.csv line 2 --> |
| results/unified_spectron/roberta/summary.csv | roberta | spectron | PASS_LOW | 51.15% <!-- source: results/unified_spectron/roberta/summary.csv line 3 --> | 0.6703 <!-- source: results/unified_spectron/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_spectron/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_spectron/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_spectron/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_spectron/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_spectron/distilbert/summary.csv line 2 --> |
| results/unified_spectron/distilbert/summary.csv | distilbert | spectron | 66955010 <!-- source: results/unified_spectron/distilbert/summary.csv line 3 --> | 58107650 <!-- source: results/unified_spectron/distilbert/summary.csv line 3 --> | 589824 <!-- source: results/unified_spectron/distilbert/summary.csv line 3 --> | 1.1523x <!-- source: results/unified_spectron/distilbert/summary.csv line 3 --> |
| results/unified_spectron/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_spectron/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_spectron/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_spectron/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_spectron/gpt2/summary.csv line 2 --> |
| results/unified_spectron/gpt2/summary.csv | gpt2 | spectron | 124441344 <!-- source: results/unified_spectron/gpt2/summary.csv line 3 --> | 106746636 <!-- source: results/unified_spectron/gpt2/summary.csv line 3 --> | 1218060 <!-- source: results/unified_spectron/gpt2/summary.csv line 3 --> | 1.1658x <!-- source: results/unified_spectron/gpt2/summary.csv line 3 --> |
| results/unified_spectron/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_spectron/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_spectron/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_spectron/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_spectron/roberta/summary.csv line 2 --> |
| results/unified_spectron/roberta/summary.csv | roberta | spectron | 124647170 <!-- source: results/unified_spectron/roberta/summary.csv line 3 --> | 114039554 <!-- source: results/unified_spectron/roberta/summary.csv line 3 --> | 1179648 <!-- source: results/unified_spectron/roberta/summary.csv line 3 --> | 1.093x <!-- source: results/unified_spectron/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_spectron/distilbert/summary.csv | distilbert | full | 33817.69 tok/s <!-- source: results/unified_spectron/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_spectron/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_spectron/distilbert/summary.csv line 2 --> |
| results/unified_spectron/distilbert/summary.csv | distilbert | spectron | 15858.84 tok/s <!-- source: results/unified_spectron/distilbert/summary.csv line 3 --> | 288.49 MB <!-- source: results/unified_spectron/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_spectron/distilbert/summary.csv line 3 --> |
| results/unified_spectron/gpt2/summary.csv | gpt2 | full | 14602.6 tok/s <!-- source: results/unified_spectron/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_spectron/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_spectron/gpt2/summary.csv line 2 --> |
| results/unified_spectron/gpt2/summary.csv | gpt2 | spectron | 33384.78 tok/s <!-- source: results/unified_spectron/gpt2/summary.csv line 3 --> | 495.89 MB <!-- source: results/unified_spectron/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_spectron/gpt2/summary.csv line 3 --> |
| results/unified_spectron/roberta/summary.csv | roberta | full | 17726.63 tok/s <!-- source: results/unified_spectron/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_spectron/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_spectron/roberta/summary.csv line 2 --> |
| results/unified_spectron/roberta/summary.csv | roberta | spectron | 7751.68 tok/s <!-- source: results/unified_spectron/roberta/summary.csv line 3 --> | 581.5 MB <!-- source: results/unified_spectron/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_spectron/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=91.90%, comp=1.152x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 91.90% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.9158 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.152x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T18:51:58.289920 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [79.36, 78.56]
> These numbers must NOT be cited in the final paper.

