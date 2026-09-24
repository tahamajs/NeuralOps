| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Factorised Matrix State** | **77.06%** | -13.42% | PASS | 0.7608 | 0 | **0.99x** | 342.5 MB | 5.4s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Factorised Matrix State** | **74.77%** | -15.83% | PASS | 0.7727 | 85 | **0.94x** | 410.3 MB | 64.5s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Factorised Matrix State** | **70.76%** | -17.08% | PASS | 0.7615 | 72 | **0.99x** | 349.5 MB | 51.1s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_factorised_matrix_state/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 2 --> |
| results/unified_factorised_matrix_state/distilbert/summary.csv | distilbert | factorised_matrix_state | PASS_NO_COMPRESSION | 76.49% <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> | 0.7491 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> |
| results/unified_factorised_matrix_state/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 2 --> |
| results/unified_factorised_matrix_state/gpt2/summary.csv | gpt2 | factorised_matrix_state | PASS_NO_COMPRESSION | 57.8% <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> | 0.6535 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> |
| results/unified_factorised_matrix_state/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 2 --> |
| results/unified_factorised_matrix_state/roberta/summary.csv | roberta | factorised_matrix_state | PASS_NO_COMPRESSION | 52.87% <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> | 0.5136 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_factorised_matrix_state/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 2 --> |
| results/unified_factorised_matrix_state/distilbert/summary.csv | distilbert | factorised_matrix_state | 66955010 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> | 67839842 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> | 884832 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> |
| results/unified_factorised_matrix_state/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 2 --> |
| results/unified_factorised_matrix_state/gpt2/summary.csv | gpt2 | factorised_matrix_state | 124441344 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> | 126211008 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> | 1769664 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> |
| results/unified_factorised_matrix_state/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 2 --> |
| results/unified_factorised_matrix_state/roberta/summary.csv | roberta | factorised_matrix_state | 124647170 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> | 133503938 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> | 1769664 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_factorised_matrix_state/distilbert/summary.csv | distilbert | full | 50938.83 tok/s <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 2 --> |
| results/unified_factorised_matrix_state/distilbert/summary.csv | distilbert | factorised_matrix_state | 15493.53 tok/s <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> | 339.99 MB <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> |
| results/unified_factorised_matrix_state/gpt2/summary.csv | gpt2 | full | 26285.77 tok/s <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 2 --> |
| results/unified_factorised_matrix_state/gpt2/summary.csv | gpt2 | factorised_matrix_state | 9921.96 tok/s <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> | 574.03 MB <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> |
| results/unified_factorised_matrix_state/roberta/summary.csv | roberta | full | 27504.61 tok/s <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 2 --> |
| results/unified_factorised_matrix_state/roberta/summary.csv | roberta | factorised_matrix_state | 7844.34 tok/s <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> | 672.35 MB <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 75.00% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-25 23:52:57 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 79.36, 78.56]
> These numbers must NOT be cited in the final paper.

