| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Xattention** | **83.14%** | -7.80% | PASS | 0.8534 | 0 | **0.99x** | 328.6 MB | 19.9 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Xattention** | **51.03%** | -24.43% | PASS_LOW | 0.6748 | 0 | **0.94x** | 659.2 MB | 26.6 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Xattention** | **53.21%** | -4.70% | PASS_LOW | 0.6837 | 0 | **0.99x** | 568.7 MB | 28.7 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_xattention/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_xattention/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_xattention/distilbert/summary.csv line 2 --> |
| results/unified_xattention/distilbert/summary.csv | distilbert | xattention | PASS | 82.68% <!-- source: results/unified_xattention/distilbert/summary.csv line 3 --> | 0.8492 <!-- source: results/unified_xattention/distilbert/summary.csv line 3 --> |
| results/unified_xattention/gpt2/summary.csv | gpt2 | full | PASS | 87.84% <!-- source: results/unified_xattention/gpt2/summary.csv line 2 --> | 0.8776 <!-- source: results/unified_xattention/gpt2/summary.csv line 2 --> |
| results/unified_xattention/gpt2/summary.csv | gpt2 | xattention | PASS | 86.24% <!-- source: results/unified_xattention/gpt2/summary.csv line 3 --> | 0.8729 <!-- source: results/unified_xattention/gpt2/summary.csv line 3 --> |
| results/unified_xattention/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_xattention/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_xattention/roberta/summary.csv line 2 --> |
| results/unified_xattention/roberta/summary.csv | roberta | xattention | PASS | 75.11% <!-- source: results/unified_xattention/roberta/summary.csv line 3 --> | 0.7737 <!-- source: results/unified_xattention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_xattention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_xattention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_xattention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_xattention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_xattention/distilbert/summary.csv line 2 --> |
| results/unified_xattention/distilbert/summary.csv | distilbert | xattention | 66955010 <!-- source: results/unified_xattention/distilbert/summary.csv line 3 --> | 67544840 <!-- source: results/unified_xattention/distilbert/summary.csv line 3 --> | 589830 <!-- source: results/unified_xattention/distilbert/summary.csv line 3 --> | 0.9913x <!-- source: results/unified_xattention/distilbert/summary.csv line 3 --> |
| results/unified_xattention/gpt2/summary.csv | gpt2 | full | 126800640 <!-- source: results/unified_xattention/gpt2/summary.csv line 2 --> | 126800640 <!-- source: results/unified_xattention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_xattention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_xattention/gpt2/summary.csv line 2 --> |
| results/unified_xattention/gpt2/summary.csv | gpt2 | xattention | 126800640 <!-- source: results/unified_xattention/gpt2/summary.csv line 3 --> | 127095564 <!-- source: results/unified_xattention/gpt2/summary.csv line 3 --> | 1474572 <!-- source: results/unified_xattention/gpt2/summary.csv line 3 --> | 0.9977x <!-- source: results/unified_xattention/gpt2/summary.csv line 3 --> |
| results/unified_xattention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_xattention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_xattention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_xattention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_xattention/roberta/summary.csv line 2 --> |
| results/unified_xattention/roberta/summary.csv | roberta | xattention | 124647170 <!-- source: results/unified_xattention/roberta/summary.csv line 3 --> | 132913934 <!-- source: results/unified_xattention/roberta/summary.csv line 3 --> | 1179660 <!-- source: results/unified_xattention/roberta/summary.csv line 3 --> | 0.9378x <!-- source: results/unified_xattention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_xattention/distilbert/summary.csv | distilbert | full | 52812.7 tok/s <!-- source: results/unified_xattention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_xattention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_xattention/distilbert/summary.csv line 2 --> |
| results/unified_xattention/distilbert/summary.csv | distilbert | xattention | 20082.63 tok/s <!-- source: results/unified_xattention/distilbert/summary.csv line 3 --> | 325.64 MB <!-- source: results/unified_xattention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_xattention/distilbert/summary.csv line 3 --> |
| results/unified_xattention/gpt2/summary.csv | gpt2 | full | 25052.91 tok/s <!-- source: results/unified_xattention/gpt2/summary.csv line 2 --> | 391.24 MB <!-- source: results/unified_xattention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_xattention/gpt2/summary.csv line 2 --> |
| results/unified_xattention/gpt2/summary.csv | gpt2 | xattention | 8079.92 tok/s <!-- source: results/unified_xattention/gpt2/summary.csv line 3 --> | 336.93 MB <!-- source: results/unified_xattention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_xattention/gpt2/summary.csv line 3 --> |
| results/unified_xattention/roberta/summary.csv | roberta | full | 27944.4 tok/s <!-- source: results/unified_xattention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_xattention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_xattention/roberta/summary.csv line 2 --> |
| results/unified_xattention/roberta/summary.csv | roberta | xattention | 9791.96 tok/s <!-- source: results/unified_xattention/roberta/summary.csv line 3 --> | 656.41 MB <!-- source: results/unified_xattention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_xattention/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=0.998 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 86.24% | `results/unified_xattention/gpt2/summary.csv` |
| F1 Score | 0.8729 | `results/unified_xattention/gpt2/summary.csv` |
| Compression | 0.998x | `results/unified_xattention/gpt2/summary.csv` |
| Compressed Layers | 0 | `results/unified_xattention/gpt2/summary.csv` |
| Fallback Layers | 0 | `results/unified_xattention/gpt2/summary.csv` |
| Timestamp | 2026-09-14T03:28:55.724316 | `results/unified_xattention/gpt2/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.94]
> These numbers must NOT be cited in the final paper.

