| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Modegpt** | **90.48%** | +0.00% | PASS | 0.9077 | 0 | **1.00x** | 311.2 MB | 17.9 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Modegpt** | **72.02%** | -3.09% | PASS | 0.7724 | 0 | **1.00x** | 301.7 MB | 25.5 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Modegpt** | **60.55%** | -0.23% | PASS | 0.6861 | 1 | **1.00x** | 639.7 MB | 25.6 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_modegpt/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_modegpt/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_modegpt/distilbert/summary.csv line 2 --> |
| results/unified_modegpt/distilbert/summary.csv | distilbert | modegpt | PASS | 90.48% <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> | 0.9077 <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> |
| results/unified_modegpt/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_modegpt/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_modegpt/gpt2/summary.csv line 2 --> |
| results/unified_modegpt/gpt2/summary.csv | gpt2 | modegpt | PASS | 60.55% <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> | 0.6861 <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> |
| results/unified_modegpt/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_modegpt/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_modegpt/roberta/summary.csv line 2 --> |
| results/unified_modegpt/roberta/summary.csv | roberta | modegpt | PASS | 72.02% <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> | 0.7724 <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | modegpt | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_modegpt/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_modegpt/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_modegpt/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_modegpt/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_modegpt/distilbert/summary.csv line 2 --> |
| results/unified_modegpt/distilbert/summary.csv | distilbert | modegpt | 66955010 <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> |
| results/unified_modegpt/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_modegpt/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_modegpt/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_modegpt/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_modegpt/gpt2/summary.csv line 2 --> |
| results/unified_modegpt/gpt2/summary.csv | gpt2 | modegpt | 124441344 <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> | 124441345 <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> | 1537 <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> |
| results/unified_modegpt/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_modegpt/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_modegpt/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_modegpt/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_modegpt/roberta/summary.csv line 2 --> |
| results/unified_modegpt/roberta/summary.csv | roberta | modegpt | 124647170 <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> | 592130 <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | modegpt | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_modegpt/distilbert/summary.csv | distilbert | full | 135712.47 tok/s <!-- source: results/unified_modegpt/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_modegpt/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_modegpt/distilbert/summary.csv line 2 --> |
| results/unified_modegpt/distilbert/summary.csv | distilbert | modegpt | 151261.61 tok/s <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> |
| results/unified_modegpt/gpt2/summary.csv | gpt2 | full | 64771.36 tok/s <!-- source: results/unified_modegpt/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_modegpt/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_modegpt/gpt2/summary.csv line 2 --> |
| results/unified_modegpt/gpt2/summary.csv | gpt2 | modegpt | 62682.11 tok/s <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> | 639.66 MB <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> | 0.420330673456192 <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> |
| results/unified_modegpt/roberta/summary.csv | roberta | full | 100571.41 tok/s <!-- source: results/unified_modegpt/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_modegpt/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_modegpt/roberta/summary.csv line 2 --> |
| results/unified_modegpt/roberta/summary.csv | roberta | modegpt | 96021.73 tok/s <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | modegpt | Elapsed 0.901s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> | 128.5 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 91.06% | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| F1 Score | 0.9137 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Fallback Layers | 36 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Timestamp | 2026-09-19T01:36:51.849707 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

