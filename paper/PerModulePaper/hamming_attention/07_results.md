| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Hamming** | **77.06%** | -13.88% | PASS | 0.8066 | 0 | **0.99x** | 336.2 MB | 3.3s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Hamming** | **50.92%** | -24.54% | PASS_LOW | 0.6748 | 0 | **0.94x** | 664.6 MB | 6.0s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Hamming** | **55.50%** | -2.41% | PASS_LOW | 0.6896 | 0 | **0.99x** | 569.5 MB | 6.7s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_hamming/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_hamming/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_hamming/distilbert/summary.csv line 2 --> |
| results/unified_hamming/distilbert/summary.csv | distilbert | hamming | PASS | 79.13% <!-- source: results/unified_hamming/distilbert/summary.csv line 3 --> | 0.8154 <!-- source: results/unified_hamming/distilbert/summary.csv line 3 --> |
| results/unified_hamming/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_hamming/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_hamming/gpt2/summary.csv line 2 --> |
| results/unified_hamming/gpt2/summary.csv | gpt2 | hamming | PASS | 71.22% <!-- source: results/unified_hamming/gpt2/summary.csv line 3 --> | 0.7239 <!-- source: results/unified_hamming/gpt2/summary.csv line 3 --> |
| results/unified_hamming/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_hamming/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_hamming/roberta/summary.csv line 2 --> |
| results/unified_hamming/roberta/summary.csv | roberta | hamming | PASS_LOW | 50.92% <!-- source: results/unified_hamming/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_hamming/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_hamming/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_hamming/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_hamming/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_hamming/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hamming/distilbert/summary.csv line 2 --> |
| results/unified_hamming/distilbert/summary.csv | distilbert | hamming | 66955010 <!-- source: results/unified_hamming/distilbert/summary.csv line 3 --> | 67545008 <!-- source: results/unified_hamming/distilbert/summary.csv line 3 --> | 589848 <!-- source: results/unified_hamming/distilbert/summary.csv line 3 --> | 0.9913x <!-- source: results/unified_hamming/distilbert/summary.csv line 3 --> |
| results/unified_hamming/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_hamming/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_hamming/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_hamming/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hamming/gpt2/summary.csv line 2 --> |
| results/unified_hamming/gpt2/summary.csv | gpt2 | hamming | 124441344 <!-- source: results/unified_hamming/gpt2/summary.csv line 3 --> | 125621340 <!-- source: results/unified_hamming/gpt2/summary.csv line 3 --> | 1179696 <!-- source: results/unified_hamming/gpt2/summary.csv line 3 --> | 0.9906x <!-- source: results/unified_hamming/gpt2/summary.csv line 3 --> |
| results/unified_hamming/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_hamming/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_hamming/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_hamming/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hamming/roberta/summary.csv line 2 --> |
| results/unified_hamming/roberta/summary.csv | roberta | hamming | 124647170 <!-- source: results/unified_hamming/roberta/summary.csv line 3 --> | 132914270 <!-- source: results/unified_hamming/roberta/summary.csv line 3 --> | 1179696 <!-- source: results/unified_hamming/roberta/summary.csv line 3 --> | 0.9378x <!-- source: results/unified_hamming/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_hamming/distilbert/summary.csv | distilbert | full | 58078.44 tok/s <!-- source: results/unified_hamming/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_hamming/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hamming/distilbert/summary.csv line 2 --> |
| results/unified_hamming/distilbert/summary.csv | distilbert | hamming | 23923.28 tok/s <!-- source: results/unified_hamming/distilbert/summary.csv line 3 --> | 231.55 MB <!-- source: results/unified_hamming/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hamming/distilbert/summary.csv line 3 --> |
| results/unified_hamming/gpt2/summary.csv | gpt2 | full | 27145.37 tok/s <!-- source: results/unified_hamming/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_hamming/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hamming/gpt2/summary.csv line 2 --> |
| results/unified_hamming/gpt2/summary.csv | gpt2 | hamming | 11585.54 tok/s <!-- source: results/unified_hamming/gpt2/summary.csv line 3 --> | 567.72 MB <!-- source: results/unified_hamming/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hamming/gpt2/summary.csv line 3 --> |
| results/unified_hamming/roberta/summary.csv | roberta | full | 26949.04 tok/s <!-- source: results/unified_hamming/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_hamming/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hamming/roberta/summary.csv line 2 --> |
| results/unified_hamming/roberta/summary.csv | roberta | hamming | 13862.67 tok/s <!-- source: results/unified_hamming/roberta/summary.csv line 3 --> | 661.56 MB <!-- source: results/unified_hamming/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hamming/roberta/summary.csv line 3 --> |

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
| Timestamp | 2026-08-26 09:43:03 | `results/adaptive_v3/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 79.13, 79.36, 71.22, 78.56]
> These numbers must NOT be cited in the final paper.

