| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Soka** | **89.91%** | -0.57% | PASS | 0.8993 | 0 | **1.00x** | 339.2 MB | 20.1 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Soka** | **84.06%** | +8.95% | PASS | 0.8494 | 0 | **1.00x** | 359.0 MB | 27.3 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Soka** | **60.78%** | +0.00% | PASS | 0.6559 | 0 | **1.00x** | 637.1 MB | 26.0 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_soka/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_soka/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_soka/distilbert/summary.csv line 2 --> |
| results/unified_soka/distilbert/summary.csv | distilbert | soka | PASS | 89.91% <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> | 0.8993 <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> |
| results/unified_soka/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_soka/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_soka/gpt2/summary.csv line 2 --> |
| results/unified_soka/gpt2/summary.csv | gpt2 | soka | PASS | 60.78% <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> | 0.6559 <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> |
| results/unified_soka/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_soka/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_soka/roberta/summary.csv line 2 --> |
| results/unified_soka/roberta/summary.csv | roberta | soka | PASS | 84.06% <!-- source: results/unified_soka/roberta/summary.csv line 3 --> | 0.8494 <!-- source: results/unified_soka/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | soka | PASS | Comp 59.91x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_soka/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_soka/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_soka/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_soka/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soka/distilbert/summary.csv line 2 --> |
| results/unified_soka/distilbert/summary.csv | distilbert | soka | 66955010 <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> |
| results/unified_soka/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_soka/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_soka/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_soka/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soka/gpt2/summary.csv line 2 --> |
| results/unified_soka/gpt2/summary.csv | gpt2 | soka | 124441344 <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> | 124048129 <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> | 198145 <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> | 1.0032x <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> |
| results/unified_soka/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_soka/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_soka/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_soka/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soka/roberta/summary.csv line 2 --> |
| results/unified_soka/roberta/summary.csv | roberta | soka | 124647170 <!-- source: results/unified_soka/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_soka/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_soka/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_soka/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | soka | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> | Reduced (59.91x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> | 59.91x <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_soka/distilbert/summary.csv | distilbert | full | 129792.67 tok/s <!-- source: results/unified_soka/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_soka/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soka/distilbert/summary.csv line 2 --> |
| results/unified_soka/distilbert/summary.csv | distilbert | soka | 139635.41 tok/s <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> | 339.22 MB <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> |
| results/unified_soka/gpt2/summary.csv | gpt2 | full | 65001.96 tok/s <!-- source: results/unified_soka/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_soka/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soka/gpt2/summary.csv line 2 --> |
| results/unified_soka/gpt2/summary.csv | gpt2 | soka | 63849.53 tok/s <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> | 637.1 MB <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> |
| results/unified_soka/roberta/summary.csv | roberta | full | 93934.25 tok/s <!-- source: results/unified_soka/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_soka/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soka/roberta/summary.csv line 2 --> |
| results/unified_soka/roberta/summary.csv | roberta | soka | 85290.05 tok/s <!-- source: results/unified_soka/roberta/summary.csv line 3 --> | 359.04 MB <!-- source: results/unified_soka/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soka/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | soka | Elapsed 0.756s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> | 136.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=60.78%, comp=1.003x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 60.78% | `results/unified_soka/gpt2/summary.csv` |
| F1 Score | 0.6559 | `results/unified_soka/gpt2/summary.csv` |
| Compression | 1.003x | `results/unified_soka/gpt2/summary.csv` |
| Compressed Layers | 0 | `results/unified_soka/gpt2/summary.csv` |
| Fallback Layers | 0 | `results/unified_soka/gpt2/summary.csv` |
| Timestamp | 2026-09-11T08:34:03.874545 | `results/unified_soka/gpt2/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

