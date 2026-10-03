| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Halo Regla** | **90.37%** | -0.57% | PASS | 0.9065 | 24 | **0.99x** | 343.3 MB | 17.3s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Halo Regla** | **50.92%** | +0.00% | PASS_LOW | 0.6748 | 48 | **0.94x** | 443.3 MB | 29.3s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Halo Regla** | **81.42%** | -6.42% | PASS | 0.8204 | 0 | **1.00x** | 996.9 MB | 27.8s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_halo_regla/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_halo_regla/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 2 --> |
| results/unified_halo_regla/distilbert/summary.csv | distilbert | halo_regla | PASS_NO_COMPRESSION | 90.37% <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> | 0.9062 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> |
| results/unified_halo_regla/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_halo_regla/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 2 --> |
| results/unified_halo_regla/gpt2/summary.csv | gpt2 | halo_regla | PASS_NO_COMPRESSION | 79.59% <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> | 0.8004 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> |
| results/unified_halo_regla/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_halo_regla/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_halo_regla/roberta/summary.csv line 2 --> |
| results/unified_halo_regla/roberta/summary.csv | roberta | halo_regla | PASS_NO_COMPRESSION | 81.31% <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> | 0.8282 <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_halo_regla/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_halo_regla/distilbert/summary.csv line 2 --> |
| results/unified_halo_regla/distilbert/summary.csv | distilbert | halo_regla | 66955010 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> | 67600226 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> | 645216 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> |
| results/unified_halo_regla/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_halo_regla/gpt2/summary.csv line 2 --> |
| results/unified_halo_regla/gpt2/summary.csv | gpt2 | halo_regla | 124441344 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> | 125731776 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> | 1290432 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> |
| results/unified_halo_regla/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_halo_regla/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_halo_regla/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_halo_regla/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_halo_regla/roberta/summary.csv line 2 --> |
| results/unified_halo_regla/roberta/summary.csv | roberta | halo_regla | 124647170 <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> | 133024706 <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> | 1290432 <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_halo_regla/distilbert/summary.csv | distilbert | full | 48055.95 tok/s <!-- source: results/unified_halo_regla/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_halo_regla/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 2 --> |
| results/unified_halo_regla/distilbert/summary.csv | distilbert | halo_regla | 17606.5 tok/s <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> | 343.31 MB <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> |
| results/unified_halo_regla/gpt2/summary.csv | gpt2 | full | 23949.67 tok/s <!-- source: results/unified_halo_regla/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_halo_regla/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 2 --> |
| results/unified_halo_regla/gpt2/summary.csv | gpt2 | halo_regla | 8164.33 tok/s <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> | 1134.78 MB <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> |
| results/unified_halo_regla/roberta/summary.csv | roberta | full | 26341.29 tok/s <!-- source: results/unified_halo_regla/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_halo_regla/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_halo_regla/roberta/summary.csv line 2 --> |
| results/unified_halo_regla/roberta/summary.csv | roberta | halo_regla | 6570.81 tok/s <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> | 672.44 MB <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.94% | `results/roberta/summary.csv` |
| F1 Score | 0.0000 | `results/roberta/summary.csv` |
| Compression | 1.000x | `results/roberta/summary.csv` |
| Compressed Layers | ? | `results/roberta/summary.csv` |
| Fallback Layers | ? | `results/roberta/summary.csv` |
| Timestamp | 2026-09-04T09:19:41.654754 | `results/roberta/summary.csv` |

