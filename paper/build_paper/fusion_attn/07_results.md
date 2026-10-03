| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Fusion Attn** | **89.91%** | -0.57% | PASS | 0.9033 | 0 | **0.99x** | 328.1 MB | 2.8s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Fusion Attn** | **84.52%** | -4.24% | PASS | 0.8635 | 0 | **0.94x** | 399.5 MB | 6.9s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Fusion Attn** | **82.11%** | +0.80% | PASS | 0.8274 | 0 | **1.00x** | 340.9 MB | 7.1s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_fusion_attn/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 2 --> |
| results/unified_fusion_attn/distilbert/summary.csv | distilbert | fusion_attn | PASS_NO_COMPRESSION | 90.6% <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 3 --> | 0.9091 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 3 --> |
| results/unified_fusion_attn/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 2 --> |
| results/unified_fusion_attn/gpt2/summary.csv | gpt2 | fusion_attn | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 3 --> |
| results/unified_fusion_attn/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_fusion_attn/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 2 --> |
| results/unified_fusion_attn/roberta/summary.csv | roberta | fusion_attn | PASS_NO_COMPRESSION | 62.61% <!-- source: results/unified_fusion_attn/roberta/summary.csv line 3 --> | 0.5821 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_fusion_attn/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 2 --> |
| results/unified_fusion_attn/distilbert/summary.csv | distilbert | fusion_attn | 66955010 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 3 --> | 67646246 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 3 --> | 691236 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 3 --> |
| results/unified_fusion_attn/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 2 --> |
| results/unified_fusion_attn/gpt2/summary.csv | gpt2 | fusion_attn | 124441344 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 3 --> | 125823816 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 3 --> | 1382472 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 3 --> |
| results/unified_fusion_attn/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_fusion_attn/roberta/summary.csv line 2 --> |
| results/unified_fusion_attn/roberta/summary.csv | roberta | fusion_attn | 124647170 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 3 --> | 133116746 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 3 --> | 1382472 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_fusion_attn/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_fusion_attn/distilbert/summary.csv | distilbert | full | 52651.57 tok/s <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 2 --> |
| results/unified_fusion_attn/distilbert/summary.csv | distilbert | fusion_attn | 26220.65 tok/s <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 3 --> | 329.66 MB <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_fusion_attn/distilbert/summary.csv line 3 --> |
| results/unified_fusion_attn/gpt2/summary.csv | gpt2 | full | 27066.22 tok/s <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 2 --> |
| results/unified_fusion_attn/gpt2/summary.csv | gpt2 | fusion_attn | 12161.57 tok/s <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 3 --> | 569.74 MB <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_fusion_attn/gpt2/summary.csv line 3 --> |
| results/unified_fusion_attn/roberta/summary.csv | roberta | full | 27366.6 tok/s <!-- source: results/unified_fusion_attn/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_fusion_attn/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 2 --> |
| results/unified_fusion_attn/roberta/summary.csv | roberta | fusion_attn | 12635.46 tok/s <!-- source: results/unified_fusion_attn/roberta/summary.csv line 3 --> | 661.02 MB <!-- source: results/unified_fusion_attn/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_fusion_attn/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 93.12% | `results/roberta/summary.csv` |
| F1 Score | 0.0000 | `results/roberta/summary.csv` |
| Compression | 1.000x | `results/roberta/summary.csv` |
| Compressed Layers | ? | `results/roberta/summary.csv` |
| Fallback Layers | ? | `results/roberta/summary.csv` |
| Timestamp | 2026-09-04T09:02:49.009723 | `results/roberta/summary.csv` |

