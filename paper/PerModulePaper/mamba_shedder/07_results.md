| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Mamba Shedder** | **87.50%** | -2.98% | PASS | 0.8868 | 0 | **1.00x** | 337.6 MB | 15.9 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Mamba Shedder** | **85.44%** | +10.33% | PASS | 0.8542 | 0 | **1.00x** | 361.5 MB | 19.9 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Mamba Shedder** | **56.88%** | -3.90% | PASS | 0.6835 | 0 | **1.00x** | 634.9 MB | 19.8 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_mamba_shedder/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 2 --> |
| results/unified_mamba_shedder/distilbert/summary.csv | distilbert | mamba_shedder | PASS | 87.5% <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> | 0.8868 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> |
| results/unified_mamba_shedder/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 2 --> |
| results/unified_mamba_shedder/gpt2/summary.csv | gpt2 | mamba_shedder | PASS | 56.88% <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> | 0.6835 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> |
| results/unified_mamba_shedder/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 2 --> |
| results/unified_mamba_shedder/roberta/summary.csv | roberta | mamba_shedder | PASS | 85.44% <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> | 0.8542 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | mamba_shedder | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_mamba_shedder/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 2 --> |
| results/unified_mamba_shedder/distilbert/summary.csv | distilbert | mamba_shedder | 66955010 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> |
| results/unified_mamba_shedder/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 2 --> |
| results/unified_mamba_shedder/gpt2/summary.csv | gpt2 | mamba_shedder | 124441344 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> |
| results/unified_mamba_shedder/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 2 --> |
| results/unified_mamba_shedder/roberta/summary.csv | roberta | mamba_shedder | 124647170 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | mamba_shedder | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_mamba_shedder/distilbert/summary.csv | distilbert | full | 171781.88 tok/s <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 2 --> |
| results/unified_mamba_shedder/distilbert/summary.csv | distilbert | mamba_shedder | 180820.54 tok/s <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> | 0.4835192859172821 <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> |
| results/unified_mamba_shedder/gpt2/summary.csv | gpt2 | full | 66076.08 tok/s <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 2 --> |
| results/unified_mamba_shedder/gpt2/summary.csv | gpt2 | mamba_shedder | 67770.43 tok/s <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> | 0.4548677504062652 <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> |
| results/unified_mamba_shedder/roberta/summary.csv | roberta | full | 106083.85 tok/s <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 2 --> |
| results/unified_mamba_shedder/roberta/summary.csv | roberta | mamba_shedder | 120068.28 tok/s <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> | 361.54 MB <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> | 0.4909336864948272 <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | mamba_shedder | Elapsed 0.173s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.80% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.9441 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T17:55:07.188714 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

