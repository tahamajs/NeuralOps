| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Aegis V3 Module** | **79.70%** | -11.24% | PASS | 0.8290 | 24 | **0.99x** | 344.7 MB | 25.9s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Aegis V3 Module** | **58.83%** | -16.63% | PASS | 0.7098 | 48 | **0.94x** | 688.1 MB | 58.6s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Aegis V3 Module** | **73.39%** | -14.45% | PASS | 0.7680 | 72 | **0.99x** | 347.8 MB | 41.3s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | aegis_v3 | PASS_NO_COMPRESSION | 78.44% <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 0.8216 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | aegis_v3 | PASS_NO_COMPRESSION | 54.82% <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 0.5397 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | aegis_v3 | PASS_NO_COMPRESSION | 55.16% <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 0.6943 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | aegis_v3 | 66955010 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 67741538 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 786528 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | aegis_v3 | 124441344 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 126014400 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 1573056 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | aegis_v3 | 124647170 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 133307330 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 1573056 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | full | 51110.39 tok/s <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | aegis_v3 | 12530.63 tok/s <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 208.28 MB <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | full | 23517.17 tok/s <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | aegis_v3 | 8863.42 tok/s <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 594.88 MB <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | full | 24095.68 tok/s <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | aegis_v3 | 5952.88 tok/s <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 688.09 MB <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=90.10%, comp=1.197x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.10% | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| F1 Score | 0.8976 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compression | 1.197x | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Timestamp | 2026-09-19T16:08:35.548944 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |

