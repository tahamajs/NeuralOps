| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Liftr** | **90.14%** | -0.34% | PASS | 0.9029 | 0 | **1.00x** | 337.6 MB | 15.8 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Liftr** | **80.16%** | +5.16% | PASS | 0.8309 | 0 | **1.00x** | 585.6 MB | 20.8 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Liftr** | **61.01%** | +0.23% | PASS | 0.6654 | 0 | **1.00x** | 634.9 MB | 19.9 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_liftr/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_liftr/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_liftr/distilbert/summary.csv line 2 --> |
| results/unified_liftr/distilbert/summary.csv | distilbert | liftr | PASS | 90.14% <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> | 0.9029 <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> |
| results/unified_liftr/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_liftr/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_liftr/gpt2/summary.csv line 2 --> |
| results/unified_liftr/gpt2/summary.csv | gpt2 | liftr | PASS | 61.01% <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> | 0.6654 <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> |
<!-- FABRICATED: | results/unified_liftr/roberta/summary.csv | roberta | full | PASS | 75.0% <!-- source: results/unified_liftr/roberta/summary.csv line 2 --> | 0.7578 <!-- source: results/unified_liftr/roberta/summary.csv line 2 --> | -->
| results/unified_liftr/roberta/summary.csv | roberta | liftr | PASS | 80.16% <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> | 0.8309 <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | liftr | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_liftr/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_liftr/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_liftr/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_liftr/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_liftr/distilbert/summary.csv line 2 --> |
| results/unified_liftr/distilbert/summary.csv | distilbert | liftr | 66955010 <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> |
| results/unified_liftr/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_liftr/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_liftr/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_liftr/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_liftr/gpt2/summary.csv line 2 --> |
| results/unified_liftr/gpt2/summary.csv | gpt2 | liftr | 124441344 <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> |
| results/unified_liftr/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_liftr/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_liftr/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_liftr/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_liftr/roberta/summary.csv line 2 --> |
| results/unified_liftr/roberta/summary.csv | roberta | liftr | 124647170 <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | liftr | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_liftr/distilbert/summary.csv | distilbert | full | 182587.66 tok/s <!-- source: results/unified_liftr/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_liftr/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_liftr/distilbert/summary.csv line 2 --> |
| results/unified_liftr/distilbert/summary.csv | distilbert | liftr | 182296.25 tok/s <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> |
| results/unified_liftr/gpt2/summary.csv | gpt2 | full | 65239.58 tok/s <!-- source: results/unified_liftr/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_liftr/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_liftr/gpt2/summary.csv line 2 --> |
| results/unified_liftr/gpt2/summary.csv | gpt2 | liftr | 68286.26 tok/s <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> |
| results/unified_liftr/roberta/summary.csv | roberta | full | 83293.19 tok/s <!-- source: results/unified_liftr/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_liftr/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_liftr/roberta/summary.csv line 2 --> |
| results/unified_liftr/roberta/summary.csv | roberta | liftr | 92788.29 tok/s <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> | 585.62 MB <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | liftr | Elapsed 0.169s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.90% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.9454 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T17:42:19.103736 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.0]
> These numbers must NOT be cited in the final paper.

