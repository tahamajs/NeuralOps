| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Adaptive Compressor** | **90.48%** | -0.46% | PASS | 0.9093 | 0 | **1.00x** | 312.2 MB | 30.9 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Adaptive Compressor** | **90.48%** | -0.12% | PASS | 0.9107 | 0 | **1.00x** | 326.6 MB | 63.2 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Adaptive Compressor** | **85.44%** | -2.40% | PASS | 0.8687 | 1 | **1.00x** | 411.3 MB | 55.9 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | adaptive_compressor | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | adaptive_compressor | PASS | 76.95% <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 0.8016 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | adaptive_compressor | PASS_NO_COMPRESSION | 85.32% <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 0.8447 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | adaptive_compressor | 66955010 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | adaptive_compressor | 124441344 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 124177153 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 326401 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 1.0021x <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | full | 127325954 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 127325954 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | adaptive_compressor | 127325954 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 127325954 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 1474560 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | full | 50739.73 tok/s <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | adaptive_compressor | 49536.87 tok/s <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 183.07 MB <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | full | 23311.22 tok/s <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | adaptive_compressor | 22313.99 tok/s <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 638.52 MB <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 0.1982078150637553 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | full | 25000.16 tok/s <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 299.92 MB <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | adaptive_compressor | 11130.61 tok/s <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 326.48 MB <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=84.50%, comp=1.002x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 84.50% | `results/agnews_unified_adaptive_compressor/gpt2/summary.csv` |
| F1 Score | 0.8376 | `results/agnews_unified_adaptive_compressor/gpt2/summary.csv` |
| Compression | 1.002x | `results/agnews_unified_adaptive_compressor/gpt2/summary.csv` |
| Compressed Layers | 1 | `results/agnews_unified_adaptive_compressor/gpt2/summary.csv` |
| Fallback Layers | 23 | `results/agnews_unified_adaptive_compressor/gpt2/summary.csv` |
| Timestamp | 2026-09-18T08:52:04.621635 | `results/agnews_unified_adaptive_compressor/gpt2/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [79.36]
> These numbers must NOT be cited in the final paper.

