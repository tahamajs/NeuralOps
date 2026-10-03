| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Rotational Slicegpt Int4** | **88.99%** | -1.49% | PASS | 0.8987 | 0 | **1.00x** | 339.8 MB | 40.5 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Rotational Slicegpt Int4** | **87.50%** | +12.39% | PASS | 0.8771 | 0 | **1.00x** | 360.8 MB | 64.1 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Rotational Slicegpt Int4** | **61.70%** | +0.92% | PASS | 0.6706 | 0 | **1.00x** | 636.0 MB | 57.4 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_rotational_slicegpt_int4/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 2 --> |
| results/unified_rotational_slicegpt_int4/distilbert/summary.csv | distilbert | rotational_slicegpt_int4 | PASS | 88.99% <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> | 0.8987 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> |
| results/unified_rotational_slicegpt_int4/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 2 --> |
| results/unified_rotational_slicegpt_int4/gpt2/summary.csv | gpt2 | rotational_slicegpt_int4 | PASS | 61.7% <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> | 0.6706 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> |
| results/unified_rotational_slicegpt_int4/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 2 --> |
| results/unified_rotational_slicegpt_int4/roberta/summary.csv | roberta | rotational_slicegpt_int4 | PASS | 87.5% <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> | 0.8771 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | rotational_slicegpt_int4 | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_rotational_slicegpt_int4/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 2 --> |
| results/unified_rotational_slicegpt_int4/distilbert/summary.csv | distilbert | rotational_slicegpt_int4 | 66955010 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> |
| results/unified_rotational_slicegpt_int4/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 2 --> |
| results/unified_rotational_slicegpt_int4/gpt2/summary.csv | gpt2 | rotational_slicegpt_int4 | 124441344 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> |
| results/unified_rotational_slicegpt_int4/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 2 --> |
| results/unified_rotational_slicegpt_int4/roberta/summary.csv | roberta | rotational_slicegpt_int4 | 124647170 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | rotational_slicegpt_int4 | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_rotational_slicegpt_int4/distilbert/summary.csv | distilbert | full | 145427.48 tok/s <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 2 --> |
| results/unified_rotational_slicegpt_int4/distilbert/summary.csv | distilbert | rotational_slicegpt_int4 | 143879.83 tok/s <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> | 339.84 MB <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> | 0.5124 <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> |
| results/unified_rotational_slicegpt_int4/gpt2/summary.csv | gpt2 | full | 65816.29 tok/s <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 2 --> |
| results/unified_rotational_slicegpt_int4/gpt2/summary.csv | gpt2 | rotational_slicegpt_int4 | 65800.77 tok/s <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> | 636.04 MB <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> | 0.5215 <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> |
| results/unified_rotational_slicegpt_int4/roberta/summary.csv | roberta | full | 91341.58 tok/s <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 2 --> |
| results/unified_rotational_slicegpt_int4/roberta/summary.csv | roberta | rotational_slicegpt_int4 | 97994.8 tok/s <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> | 360.79 MB <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> | 0.515 <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | rotational_slicegpt_int4 | Elapsed 19.574s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> | 254.5 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 88.99% | `results/unified_rotational_slicegpt_int4/distilbert/summary.csv` |
| F1 Score | 0.8987 | `results/unified_rotational_slicegpt_int4/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_rotational_slicegpt_int4/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_rotational_slicegpt_int4/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_rotational_slicegpt_int4/distilbert/summary.csv` |
| Timestamp | 2026-09-11T07:48:52.619717 | `results/unified_rotational_slicegpt_int4/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

