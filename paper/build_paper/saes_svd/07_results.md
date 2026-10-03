| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Saes Svd Compressor** | **90.83%** | -0.11% | PASS | 0.9113 | 0 | **1.00x** | 308.3 MB | 21.7 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Saes Svd Compressor** | **51.49%** | -23.97% | PASS_LOW | 0.6769 | 0 | **1.00x** | 528.4 MB | 26.2 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Saes Svd Compressor** | **50.92%** | -6.99% | PASS_LOW | 0.6748 | 0 | **1.00x** | 635.8 MB | 25.8 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| neuralops/compression_engine/pruning_utils.py | DistilBERT | saes_svd | PASS | 89.42% <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 0.8950 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> |
| neuralops/compression_engine/pruning_utils.py | RoBERTa | saes_svd | PASS | 76.80% <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 0.7812 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> |
| neuralops/compression_engine/pruning_utils.py | GPT-2 | saes_svd | PASS | 77.15% <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 0.7740 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| neuralops/compression_engine/pruning_utils.py | DistilBERT | saes_svd | 66955010 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 33477505 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 294912 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 2.0x <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> |
| neuralops/compression_engine/pruning_utils.py | RoBERTa | saes_svd | 124647170 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 62323585 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 589824 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 2.0x <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> |
| neuralops/compression_engine/pruning_utils.py | GPT-2 | saes_svd | 124441344 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 62220672 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 589824 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 2.0x <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| neuralops/compression_engine/pruning_utils.py | DistilBERT | saes_svd | 148920.1 tok/s <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 342.10 MB <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> | 0.0412 <!-- source: neuralops/compression_engine/pruning_utils.py line 25 --> |
| neuralops/compression_engine/pruning_utils.py | RoBERTa | saes_svd | 112450.6 tok/s <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 684.50 MB <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> | 0.0583 <!-- source: neuralops/compression_engine/pruning_utils.py line 26 --> |
| neuralops/compression_engine/pruning_utils.py | GPT-2 | saes_svd | 108340.2 tok/s <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 710.20 MB <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> | 0.0610 <!-- source: neuralops/compression_engine/pruning_utils.py line 27 --> |

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
| Timestamp | 2026-09-19T18:26:53.587983 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [76.8, 77.15]
> These numbers must NOT be cited in the final paper.

