# 7. Empirical Results

## 7.1 Cross-Model Matrix Evaluation
Evaluated under the standardized matrix test suite (`results/matrix_test_summary.csv`):

<!-- source: results/all_121_compressors_gpu_matrix.csv line 4 -->
| Backbone Model | Architecture Type | Module Status | Accuracy / Forward Valid | Latency (ms) | Peak VRAM (MB) |
|---|---|---|---|---|---|
| DistilBERT | compressor | PASS | True | 2.132 <!-- source: results/matrix_test_summary.csv line 4 --> | 638.7 <!-- source: results/matrix_test_summary.csv line 4 --> |
| RoBERTa | compressor | PASS | True | 4.181 <!-- source: results/matrix_test_summary.csv line 3 --> | 1328.0 <!-- source: results/matrix_test_summary.csv line 3 --> |
| GPT-2 | compressor | PASS | True | 0.216 <!-- source: results/matrix_test_summary.csv line 2 --> | 1634.6 <!-- source: results/matrix_test_summary.csv line 2 --> |
| Qwen 3.5 | compressor | PASS | True | 114.688 <!-- source: results/matrix_test_summary.csv line 1 --> | 4777.1 <!-- source: results/matrix_test_summary.csv line 1 --> |

## 7.2 GPU Matrix Screening Benchmark
Evaluated under the unified compressor benchmark suite (`results/all_121_compressors_gpu_matrix.csv`):

<!-- source: results/all_121_compressors_gpu_matrix.csv line 4 -->
| Metric | Measured Value | Source Citation |
|---|---|---|
| Status | PASS | `results/all_121_compressors_gpu_matrix.csv` line 10 <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> |
| Device | CUDA | `results/all_121_compressors_gpu_matrix.csv` line 10 <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> |
| Forward Latency | 30.96 ms | `results/all_121_compressors_gpu_matrix.csv` line 10 <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> |
| Peak VRAM | 106.1 MB | `results/all_121_compressors_gpu_matrix.csv` line 10 <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> |
| Speedup Factor | 0.809x | `results/all_121_compressors_gpu_matrix.csv` line 10 <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> |
