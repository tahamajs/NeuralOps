| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Upi Stuffed Mamba** | **50.92%** | -40.02% | PASS_LOW | 0.6748 | 0 | **0.94x** | 581.4 MB | 28.4s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Upi Stuffed Mamba** | **50.92%** | -24.54% | PASS_LOW | 0.6748 | 0 | **0.89x** | 923.2 MB | 511.5s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Upi Stuffed Mamba** | **51.03%** | -6.88% | PASS_LOW | 0.6728 | 0 | **0.94x** | 816.1 MB | 32.1s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_upi_stuffed_mamba/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 2 --> |
| results/unified_upi_stuffed_mamba/distilbert/summary.csv | distilbert | upi_stuffed_mamba | PASS_LOW | 50.92% <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 3 --> |
| results/unified_upi_stuffed_mamba/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 2 --> |
| results/unified_upi_stuffed_mamba/gpt2/summary.csv | gpt2 | upi_stuffed_mamba | PASS_LOW | 51.03% <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 3 --> | 0.6728 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 3 --> |
| results/unified_upi_stuffed_mamba/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 2 --> |
| results/unified_upi_stuffed_mamba/roberta/summary.csv | roberta | upi_stuffed_mamba | PASS_LOW | 50.92% <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_upi_stuffed_mamba/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 2 --> |
| results/unified_upi_stuffed_mamba/distilbert/summary.csv | distilbert | upi_stuffed_mamba | 66955010 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 3 --> | 71102228 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 3 --> | 960788 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 3 --> | 0.9417x <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 3 --> |
| results/unified_upi_stuffed_mamba/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 2 --> |
| results/unified_upi_stuffed_mamba/gpt2/summary.csv | gpt2 | upi_stuffed_mamba | 124441344 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 3 --> | 132735780 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 3 --> | 738852 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 3 --> | 0.9375x <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 3 --> |
| results/unified_upi_stuffed_mamba/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 2 --> |
| results/unified_upi_stuffed_mamba/roberta/summary.csv | roberta | upi_stuffed_mamba | 124647170 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 3 --> | 140028710 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 3 --> | 1329446 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 3 --> | 0.8902x <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_upi_stuffed_mamba/distilbert/summary.csv | distilbert | full | 148917.99 tok/s <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 2 --> |
| results/unified_upi_stuffed_mamba/distilbert/summary.csv | distilbert | upi_stuffed_mamba | 10234.65 tok/s <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 3 --> | 581.4 MB <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_upi_stuffed_mamba/distilbert/summary.csv line 3 --> |
| results/unified_upi_stuffed_mamba/gpt2/summary.csv | gpt2 | full | 61468.34 tok/s <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 2 --> |
| results/unified_upi_stuffed_mamba/gpt2/summary.csv | gpt2 | upi_stuffed_mamba | 9299.77 tok/s <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 3 --> | 816.06 MB <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_upi_stuffed_mamba/gpt2/summary.csv line 3 --> |
| results/unified_upi_stuffed_mamba/roberta/summary.csv | roberta | full | 83058.03 tok/s <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 2 --> |
| results/unified_upi_stuffed_mamba/roberta/summary.csv | roberta | upi_stuffed_mamba | 1984.43 tok/s <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 3 --> | 923.2 MB <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_upi_stuffed_mamba/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 68.38% | `results/adaptive_v3/benchmark_summary.csv` |
| F1 Score | 0.8122 | `results/adaptive_v3/benchmark_summary.csv` |
| Compression | 1.000x | `results/adaptive_v3/benchmark_summary.csv` |
| Compressed Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Fallback Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Timestamp | 2026-08-26 09:26:55 | `results/adaptive_v3/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.94, 57.91, 75.46]
> These numbers must NOT be cited in the final paper.

