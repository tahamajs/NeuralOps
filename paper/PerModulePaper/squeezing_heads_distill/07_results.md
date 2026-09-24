| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Squeezing Heads Distill** | **90.37%** | -0.11% | PASS | 0.9075 | 0 | **0.99x** | 332.5 MB | 4.3s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Squeezing Heads Distill** | **51.49%** | -23.62% | PASS_LOW | 0.6754 | 0 | **0.94x** | 379.9 MB | 7.2s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Squeezing Heads Distill** | **51.61%** | -9.17% | PASS_LOW | 0.6774 | 0 | **0.99x** | 568.7 MB | 8.5s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_squeezing_heads_distill/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 2 --> |
| results/unified_squeezing_heads_distill/distilbert/summary.csv | distilbert | squeezing_heads_distill | PASS | 90.37% <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 3 --> | 0.9075 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 3 --> |
| results/unified_squeezing_heads_distill/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 2 --> |
| results/unified_squeezing_heads_distill/gpt2/summary.csv | gpt2 | squeezing_heads_distill | PASS_LOW | 51.61% <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 3 --> | 0.6774 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 3 --> |
| results/unified_squeezing_heads_distill/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 2 --> |
| results/unified_squeezing_heads_distill/roberta/summary.csv | roberta | squeezing_heads_distill | PASS_LOW | 51.49% <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 3 --> | 0.6754 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_squeezing_heads_distill/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 2 --> |
| results/unified_squeezing_heads_distill/distilbert/summary.csv | distilbert | squeezing_heads_distill | 66955010 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 3 --> | 67544858 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 3 --> | 1181978 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 3 --> | 0.9913x <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 3 --> |
| results/unified_squeezing_heads_distill/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 2 --> |
| results/unified_squeezing_heads_distill/gpt2/summary.csv | gpt2 | squeezing_heads_distill | 124441344 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 3 --> | 125621040 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 3 --> | 1181232 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 3 --> | 0.9906x <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 3 --> |
| results/unified_squeezing_heads_distill/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 2 --> |
| results/unified_squeezing_heads_distill/roberta/summary.csv | roberta | squeezing_heads_distill | 124647170 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 3 --> | 132913970 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 3 --> | 1771826 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 3 --> | 0.9378x <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_squeezing_heads_distill/distilbert/summary.csv | distilbert | full | 145913.26 tok/s <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 2 --> |
| results/unified_squeezing_heads_distill/distilbert/summary.csv | distilbert | squeezing_heads_distill | 71868.72 tok/s <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 3 --> | 332.46 MB <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_squeezing_heads_distill/distilbert/summary.csv line 3 --> |
| results/unified_squeezing_heads_distill/gpt2/summary.csv | gpt2 | full | 63601.47 tok/s <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 2 --> |
| results/unified_squeezing_heads_distill/gpt2/summary.csv | gpt2 | squeezing_heads_distill | 35939.75 tok/s <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 3 --> | 568.71 MB <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_squeezing_heads_distill/gpt2/summary.csv line 3 --> |
| results/unified_squeezing_heads_distill/roberta/summary.csv | roberta | full | 91092.48 tok/s <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 2 --> |
| results/unified_squeezing_heads_distill/roberta/summary.csv | roberta | squeezing_heads_distill | 46530.27 tok/s <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 3 --> | 379.87 MB <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_squeezing_heads_distill/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=0.991 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.37% | `results/unified_squeezing_heads_distill/distilbert/summary.csv` |
| F1 Score | 0.9075 | `results/unified_squeezing_heads_distill/distilbert/summary.csv` |
| Compression | 0.991x | `results/unified_squeezing_heads_distill/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_squeezing_heads_distill/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_squeezing_heads_distill/distilbert/summary.csv` |
| Timestamp | 2026-09-11T08:45:26.012457 | `results/unified_squeezing_heads_distill/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [60.78, 75.11]
> These numbers must NOT be cited in the final paper.

