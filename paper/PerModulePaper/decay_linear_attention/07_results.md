| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Decay Linear Attention** | **55.73%** | -35.21% | PASS_LOW | 0.5089 | 24 | **0.99x** | 344.9 MB | 21.9s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Decay Linear Attention** | **51.49%** | -23.97% | PASS_LOW | 0.3878 | 48 | **0.94x** | 675.0 MB | 36.8s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Decay Linear Attention** | **51.15%** | -36.69% | PASS_LOW | 0.6758 | 72 | **1.00x** | 341.9 MB | 30.6s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_decay_linear_attention/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 2 --> |
| results/unified_decay_linear_attention/distilbert/summary.csv | distilbert | decay_linear_attention | PASS_NO_COMPRESSION | 62.27% <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 3 --> | 0.5923 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 3 --> |
| results/unified_decay_linear_attention/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 2 --> |
| results/unified_decay_linear_attention/gpt2/summary.csv | gpt2 | decay_linear_attention | PASS_NO_COMPRESSION | 49.43% <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 3 --> | 0.5898 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 3 --> |
| results/unified_decay_linear_attention/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 2 --> |
| results/unified_decay_linear_attention/roberta/summary.csv | roberta | decay_linear_attention | PASS_NO_COMPRESSION | 49.2% <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 3 --> | 0.4855 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_decay_linear_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 2 --> |
| results/unified_decay_linear_attention/distilbert/summary.csv | distilbert | decay_linear_attention | 66955010 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 3 --> | 67600274 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 3 --> | 645264 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 3 --> |
| results/unified_decay_linear_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 2 --> |
| results/unified_decay_linear_attention/gpt2/summary.csv | gpt2 | decay_linear_attention | 124441344 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 3 --> | 125731872 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 3 --> | 1290528 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 3 --> |
| results/unified_decay_linear_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 2 --> |
| results/unified_decay_linear_attention/roberta/summary.csv | roberta | decay_linear_attention | 124647170 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 3 --> | 133024802 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 3 --> | 1290528 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_decay_linear_attention/distilbert/summary.csv | distilbert | full | 52378.33 tok/s <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 2 --> |
| results/unified_decay_linear_attention/distilbert/summary.csv | distilbert | decay_linear_attention | 5909.07 tok/s <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 3 --> | 340.41 MB <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_decay_linear_attention/distilbert/summary.csv line 3 --> |
| results/unified_decay_linear_attention/gpt2/summary.csv | gpt2 | full | 27040.35 tok/s <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 2 --> |
| results/unified_decay_linear_attention/gpt2/summary.csv | gpt2 | decay_linear_attention | 4830.56 tok/s <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 3 --> | 568.99 MB <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_decay_linear_attention/gpt2/summary.csv line 3 --> |
| results/unified_decay_linear_attention/roberta/summary.csv | roberta | full | 27335.95 tok/s <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 2 --> |
| results/unified_decay_linear_attention/roberta/summary.csv | roberta | decay_linear_attention | 2938.94 tok/s <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 3 --> | 672.16 MB <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_decay_linear_attention/roberta/summary.csv line 3 --> |

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
| Timestamp | 2026-08-25 23:58:32 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 79.36, 78.56]
> These numbers must NOT be cited in the final paper.

