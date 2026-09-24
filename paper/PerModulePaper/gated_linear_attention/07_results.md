| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Gated Linear Attention** | **50.92%** | -39.56% | PASS_LOW | 0.6748 | 0 | **0.94x** | 368.2 MB | 22.8s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Gated Linear Attention** | **50.92%** | -24.08% | PASS_LOW | 0.6748 | 0 | **0.89x** | 714.7 MB | 43.4s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Gated Linear Attention** | **50.92%** | -9.86% | PASS_LOW | 0.6748 | 0 | **0.94x** | 607.0 MB | 42.9s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_gated_linear_attention/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 2 --> |
| results/unified_gated_linear_attention/distilbert/summary.csv | distilbert | gated_linear_attention | PASS_LOW | 50.92% <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> |
| results/unified_gated_linear_attention/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 2 --> |
| results/unified_gated_linear_attention/gpt2/summary.csv | gpt2 | gated_linear_attention | PASS_LOW | 50.92% <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> |
<!-- FABRICATED: | results/unified_gated_linear_attention/roberta/summary.csv | roberta | full | PASS | 75.0% <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 2 --> | 0.7578 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 2 --> | -->
| results/unified_gated_linear_attention/roberta/summary.csv | roberta | gated_linear_attention | PASS_LOW | 50.92% <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_gated_linear_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 2 --> |
| results/unified_gated_linear_attention/distilbert/summary.csv | distilbert | gated_linear_attention | 66955010 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> | 71273582 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> | 1384814 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> | 0.9394x <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> |
| results/unified_gated_linear_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 2 --> |
| results/unified_gated_linear_attention/gpt2/summary.csv | gpt2 | gated_linear_attention | 124441344 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> | 133078488 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> | 1586904 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> | 0.9351x <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> |
| results/unified_gated_linear_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 2 --> |
| results/unified_gated_linear_attention/roberta/summary.csv | roberta | gated_linear_attention | 124647170 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> | 140371418 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> | 2177498 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> | 0.888x <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_gated_linear_attention/distilbert/summary.csv | distilbert | full | 44311.15 tok/s <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 2 --> |
| results/unified_gated_linear_attention/distilbert/summary.csv | distilbert | gated_linear_attention | 14557.05 tok/s <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> | 368.25 MB <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> |
| results/unified_gated_linear_attention/gpt2/summary.csv | gpt2 | full | 27486.42 tok/s <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 2 --> |
| results/unified_gated_linear_attention/gpt2/summary.csv | gpt2 | gated_linear_attention | 7082.21 tok/s <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> | 607.03 MB <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> |
| results/unified_gated_linear_attention/roberta/summary.csv | roberta | full | 24605.55 tok/s <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 2 --> |
| results/unified_gated_linear_attention/roberta/summary.csv | roberta | gated_linear_attention | 7769.12 tok/s <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> | 714.69 MB <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 62.50% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-25 23:55:55 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.48, 75.0]
> These numbers must NOT be cited in the final paper.

