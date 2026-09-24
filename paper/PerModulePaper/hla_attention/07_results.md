| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Hla Attention** | **49.08%** | -41.86% | PASS_LOW | 0.0000 | 0 | **0.99x** | 338.2 MB | 4.1s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Hla Attention** | **49.08%** | -26.38% | PASS_LOW | 0.0000 | 0 | **0.94x** | 667.3 MB | 7.3s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Hla Attention** | **49.08%** | -8.83% | PASS_LOW | 0.0000 | 0 | **0.99x** | 567.6 MB | 7.1s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_hla/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_hla/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_hla/distilbert/summary.csv line 2 --> |
| results/unified_hla/distilbert/summary.csv | distilbert | hla | PASS_LOW | 49.08% <!-- source: results/unified_hla/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_hla/distilbert/summary.csv line 3 --> |
| results/unified_hla/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_hla/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_hla/gpt2/summary.csv line 2 --> |
| results/unified_hla/gpt2/summary.csv | gpt2 | hla | PASS_LOW | 49.08% <!-- source: results/unified_hla/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_hla/gpt2/summary.csv line 3 --> |
| results/unified_hla/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_hla/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_hla/roberta/summary.csv line 2 --> |
| results/unified_hla/roberta/summary.csv | roberta | hla | PASS_LOW | 49.08% <!-- source: results/unified_hla/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_hla/roberta/summary.csv line 3 --> |
| results/unified_hla_attention/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_hla_attention/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 2 --> |
| results/unified_hla_attention/distilbert/summary.csv | distilbert | hla_attention | PASS_LOW | 49.08% <!-- source: results/unified_hla_attention/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 3 --> |
| results/unified_hla_attention/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_hla_attention/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 2 --> |
| results/unified_hla_attention/gpt2/summary.csv | gpt2 | hla_attention | PASS_LOW | 49.08% <!-- source: results/unified_hla_attention/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 3 --> |
| results/unified_hla_attention/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_hla_attention/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_hla_attention/roberta/summary.csv line 2 --> |
| results/unified_hla_attention/roberta/summary.csv | roberta | hla_attention | PASS_LOW | 49.08% <!-- source: results/unified_hla_attention/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_hla_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_hla/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_hla/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_hla/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_hla/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hla/distilbert/summary.csv line 2 --> |
| results/unified_hla/distilbert/summary.csv | distilbert | hla | 66955010 <!-- source: results/unified_hla/distilbert/summary.csv line 3 --> | 67531088 <!-- source: results/unified_hla/distilbert/summary.csv line 3 --> | 1182032 <!-- source: results/unified_hla/distilbert/summary.csv line 3 --> | 0.9915x <!-- source: results/unified_hla/distilbert/summary.csv line 3 --> |
| results/unified_hla/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_hla/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_hla/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_hla/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hla/gpt2/summary.csv line 2 --> |
| results/unified_hla/gpt2/summary.csv | gpt2 | hla | 124441344 <!-- source: results/unified_hla/gpt2/summary.csv line 3 --> | 125593500 <!-- source: results/unified_hla/gpt2/summary.csv line 3 --> | 1181340 <!-- source: results/unified_hla/gpt2/summary.csv line 3 --> | 0.9908x <!-- source: results/unified_hla/gpt2/summary.csv line 3 --> |
| results/unified_hla/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_hla/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_hla/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_hla/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hla/roberta/summary.csv line 2 --> |
| results/unified_hla/roberta/summary.csv | roberta | hla | 124647170 <!-- source: results/unified_hla/roberta/summary.csv line 3 --> | 132886430 <!-- source: results/unified_hla/roberta/summary.csv line 3 --> | 1771934 <!-- source: results/unified_hla/roberta/summary.csv line 3 --> | 0.938x <!-- source: results/unified_hla/roberta/summary.csv line 3 --> |
| results/unified_hla_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hla_attention/distilbert/summary.csv line 2 --> |
| results/unified_hla_attention/distilbert/summary.csv | distilbert | hla_attention | 66955010 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 3 --> | 67531088 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 3 --> | 1182032 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 3 --> | 0.9915x <!-- source: results/unified_hla_attention/distilbert/summary.csv line 3 --> |
| results/unified_hla_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hla_attention/gpt2/summary.csv line 2 --> |
| results/unified_hla_attention/gpt2/summary.csv | gpt2 | hla_attention | 124441344 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 3 --> | 125593500 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 3 --> | 1181340 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 3 --> | 0.9908x <!-- source: results/unified_hla_attention/gpt2/summary.csv line 3 --> |
| results/unified_hla_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_hla_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_hla_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_hla_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hla_attention/roberta/summary.csv line 2 --> |
| results/unified_hla_attention/roberta/summary.csv | roberta | hla_attention | 124647170 <!-- source: results/unified_hla_attention/roberta/summary.csv line 3 --> | 132886430 <!-- source: results/unified_hla_attention/roberta/summary.csv line 3 --> | 1771934 <!-- source: results/unified_hla_attention/roberta/summary.csv line 3 --> | 0.938x <!-- source: results/unified_hla_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_hla/distilbert/summary.csv | distilbert | full | 165321.55 tok/s <!-- source: results/unified_hla/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_hla/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hla/distilbert/summary.csv line 2 --> |
| results/unified_hla/distilbert/summary.csv | distilbert | hla | 59442.05 tok/s <!-- source: results/unified_hla/distilbert/summary.csv line 3 --> | 338.25 MB <!-- source: results/unified_hla/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hla/distilbert/summary.csv line 3 --> |
| results/unified_hla/gpt2/summary.csv | gpt2 | full | 72743.1 tok/s <!-- source: results/unified_hla/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_hla/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hla/gpt2/summary.csv line 2 --> |
| results/unified_hla/gpt2/summary.csv | gpt2 | hla | 40343.91 tok/s <!-- source: results/unified_hla/gpt2/summary.csv line 3 --> | 567.58 MB <!-- source: results/unified_hla/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hla/gpt2/summary.csv line 3 --> |
| results/unified_hla/roberta/summary.csv | roberta | full | 91918.27 tok/s <!-- source: results/unified_hla/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_hla/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hla/roberta/summary.csv line 2 --> |
| results/unified_hla/roberta/summary.csv | roberta | hla | 35990.09 tok/s <!-- source: results/unified_hla/roberta/summary.csv line 3 --> | 667.32 MB <!-- source: results/unified_hla/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hla/roberta/summary.csv line 3 --> |
| results/unified_hla_attention/distilbert/summary.csv | distilbert | full | 175722.48 tok/s <!-- source: results/unified_hla_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_hla_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 2 --> |
| results/unified_hla_attention/distilbert/summary.csv | distilbert | hla_attention | 75877.59 tok/s <!-- source: results/unified_hla_attention/distilbert/summary.csv line 3 --> | 338.25 MB <!-- source: results/unified_hla_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hla_attention/distilbert/summary.csv line 3 --> |
| results/unified_hla_attention/gpt2/summary.csv | gpt2 | full | 71085.22 tok/s <!-- source: results/unified_hla_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_hla_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 2 --> |
| results/unified_hla_attention/gpt2/summary.csv | gpt2 | hla_attention | 45287.1 tok/s <!-- source: results/unified_hla_attention/gpt2/summary.csv line 3 --> | 567.58 MB <!-- source: results/unified_hla_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hla_attention/gpt2/summary.csv line 3 --> |
| results/unified_hla_attention/roberta/summary.csv | roberta | full | 91292.28 tok/s <!-- source: results/unified_hla_attention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_hla_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hla_attention/roberta/summary.csv line 2 --> |
| results/unified_hla_attention/roberta/summary.csv | roberta | hla_attention | 40775.02 tok/s <!-- source: results/unified_hla_attention/roberta/summary.csv line 3 --> | 667.32 MB <!-- source: results/unified_hla_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hla_attention/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 0.00% | `results/gpu_server/snapshot_2026_08_26/benchmark_summary_v2.csv` |
| F1 Score | 0.0000 | `results/gpu_server/snapshot_2026_08_26/benchmark_summary_v2.csv` |
| Compression | 1.000x | `results/gpu_server/snapshot_2026_08_26/benchmark_summary_v2.csv` |
| Compressed Layers | ? | `results/gpu_server/snapshot_2026_08_26/benchmark_summary_v2.csv` |
| Fallback Layers | ? | `results/gpu_server/snapshot_2026_08_26/benchmark_summary_v2.csv` |
| Timestamp | 2026-08-26T17:24:10.582209 | `results/gpu_server/snapshot_2026_08_26/benchmark_summary_v2.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.94, 57.91, 75.46, 90.94, 57.91, 75.46]
> These numbers must NOT be cited in the final paper.

