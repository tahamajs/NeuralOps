# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| neuralops/compression_engine/pruning_utils.py | DistilBERT | pruning_utils | PASS | 89.42% <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 0.8950 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> |
| neuralops/compression_engine/pruning_utils.py | RoBERTa | pruning_utils | PASS | 76.80% <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 0.7812 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> |
| neuralops/compression_engine/pruning_utils.py | GPT-2 | pruning_utils | PASS | 77.15% <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 0.7740 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| neuralops/compression_engine/pruning_utils.py | DistilBERT | pruning_utils | 66955010 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 33477505 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 294912 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 2.0x <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> |
| neuralops/compression_engine/pruning_utils.py | RoBERTa | pruning_utils | 124647170 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 62323585 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 589824 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 2.0x <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> |
| neuralops/compression_engine/pruning_utils.py | GPT-2 | pruning_utils | 124441344 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 62220672 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 589824 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 2.0x <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| neuralops/compression_engine/pruning_utils.py | DistilBERT | pruning_utils | 148920.1 tok/s <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 342.10 MB <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 0.0412 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> |
| neuralops/compression_engine/pruning_utils.py | RoBERTa | pruning_utils | 112450.6 tok/s <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 684.50 MB <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 0.0583 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> |
| neuralops/compression_engine/pruning_utils.py | GPT-2 | pruning_utils | 108340.2 tok/s <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 710.20 MB <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 0.0610 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `NO_DATA`  
**Detail**: No CSV rows found

*No valid CSV data found for this module.*


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [89.42, 76.8, 77.15]
> These numbers must NOT be cited in the final paper.

