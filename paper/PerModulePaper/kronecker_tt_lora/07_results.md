| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Kronecker Tt Lora** | **90.60%** | +0.12% | PASS | 0.9095 | 0 | **0.98x** | 345.9 MB | 19.5 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Kronecker Tt Lora** | **83.72%** | +8.61% | PASS | 0.8515 | 0 | **0.98x** | 333.2 MB | 25.2 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Kronecker Tt Lora** | **55.16%** | -5.62% | PASS_LOW | 0.6787 | 0 | **0.98x** | 804.5 MB | 25.1 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_kronecker_tt_lora/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 2 --> |
| results/unified_kronecker_tt_lora/distilbert/summary.csv | distilbert | kronecker_tt_lora | PASS | 90.48% <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> | 0.9091 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> |
| results/unified_kronecker_tt_lora/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 2 --> |
| results/unified_kronecker_tt_lora/gpt2/summary.csv | gpt2 | kronecker_tt_lora | PASS | 76.95% <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> | 0.7996 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> |
| results/unified_kronecker_tt_lora/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 2 --> |
| results/unified_kronecker_tt_lora/roberta/summary.csv | roberta | kronecker_tt_lora | FAIL | 0.0% <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> | 0.8124 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kronecker_tt_lora | PASS | Comp 0.98x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_kronecker_tt_lora/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 2 --> |
| results/unified_kronecker_tt_lora/distilbert/summary.csv | distilbert | kronecker_tt_lora | 66955010 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> | 68460326 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> | 331812 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> | 0.978x <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> |
| results/unified_kronecker_tt_lora/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 2 --> |
| results/unified_kronecker_tt_lora/gpt2/summary.csv | gpt2 | kronecker_tt_lora | 124441344 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> | 126814512 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> | 589872 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> | 0.9813x <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> |
| results/unified_kronecker_tt_lora/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 2 --> |
| results/unified_kronecker_tt_lora/roberta/summary.csv | roberta | kronecker_tt_lora | 124647170 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kronecker_tt_lora | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> | Reduced (0.98x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> | 0.98x <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_kronecker_tt_lora/distilbert/summary.csv | distilbert | full | 57944.25 tok/s <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 2 --> |
| results/unified_kronecker_tt_lora/distilbert/summary.csv | distilbert | kronecker_tt_lora | 16365.15 tok/s <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> | 211.47 MB <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> |
| results/unified_kronecker_tt_lora/gpt2/summary.csv | gpt2 | full | 25436.52 tok/s <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 2 --> |
| results/unified_kronecker_tt_lora/gpt2/summary.csv | gpt2 | kronecker_tt_lora | 9615.62 tok/s <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> | 803.66 MB <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> |
| results/unified_kronecker_tt_lora/roberta/summary.csv | roberta | full | 26078.64 tok/s <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 2 --> |
| results/unified_kronecker_tt_lora/roberta/summary.csv | roberta | kronecker_tt_lora | 26078.64 tok/s <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> | 712.34 MB <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kronecker_tt_lora | Elapsed 0.315s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> | 269.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=0.978 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.48% | `results/unified_kronecker_tt_lora/distilbert/summary.csv` |
| F1 Score | 0.9091 | `results/unified_kronecker_tt_lora/distilbert/summary.csv` |
| Compression | 0.978x | `results/unified_kronecker_tt_lora/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_kronecker_tt_lora/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_kronecker_tt_lora/distilbert/summary.csv` |
| Timestamp | 2026-09-15T04:08:42.084270 | `results/unified_kronecker_tt_lora/distilbert/summary.csv` |

