# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_astro/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_astro/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_astro/distilbert/summary.csv line 2 --> |
| results/unified_astro/distilbert/summary.csv | distilbert | astro | PASS | 90.71% <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> | 0.9095 <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> |
| results/unified_astro/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_astro/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_astro/gpt2/summary.csv line 2 --> |
| results/unified_astro/gpt2/summary.csv | gpt2 | astro | PASS | 77.87% <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> | 0.7814 <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> |
| results/unified_astro/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.55% <!-- source: results/unified_astro/qwen3_5/summary.csv line 2 --> | 0.5501 <!-- source: results/unified_astro/qwen3_5/summary.csv line 2 --> |
| results/unified_astro/qwen3_5/summary.csv | qwen3_5 | astro | PASS | 87.5% <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> | 0.8659 <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> |
| results/unified_astro/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_astro/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_astro/roberta/summary.csv line 2 --> |
| results/unified_astro/roberta/summary.csv | roberta | astro | PASS | 74.77% <!-- source: results/unified_astro/roberta/summary.csv line 3 --> | 0.7936 <!-- source: results/unified_astro/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | astro | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 17 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 17 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | astro_quantizer | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 18 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 18 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_astro/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_astro/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_astro/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_astro/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_astro/distilbert/summary.csv line 2 --> |
| results/unified_astro/distilbert/summary.csv | distilbert | astro | 66955010 <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> |
| results/unified_astro/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_astro/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_astro/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_astro/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_astro/gpt2/summary.csv line 2 --> |
| results/unified_astro/gpt2/summary.csv | gpt2 | astro | 124441344 <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> |
| results/unified_astro/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_astro/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_astro/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_astro/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_astro/qwen3_5/summary.csv line 2 --> |
| results/unified_astro/qwen3_5/summary.csv | qwen3_5 | astro | 763217728 <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> | 10824704 <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> |
| results/unified_astro/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_astro/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_astro/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_astro/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_astro/roberta/summary.csv line 2 --> |
| results/unified_astro/roberta/summary.csv | roberta | astro | 124647170 <!-- source: results/unified_astro/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_astro/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_astro/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_astro/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | astro | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 17 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 17 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 17 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 17 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | astro_quantizer | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 18 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 18 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 18 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 18 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_astro/distilbert/summary.csv | distilbert | full | 51084.63 tok/s <!-- source: results/unified_astro/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_astro/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_astro/distilbert/summary.csv line 2 --> |
| results/unified_astro/distilbert/summary.csv | distilbert | astro | 51415.84 tok/s <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> | 183.07 MB <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> | 0.0375307585199194 <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> |
| results/unified_astro/gpt2/summary.csv | gpt2 | full | 24305.76 tok/s <!-- source: results/unified_astro/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_astro/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_astro/gpt2/summary.csv line 2 --> |
| results/unified_astro/gpt2/summary.csv | gpt2 | astro | 24022.36 tok/s <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> | 627.28 MB <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> | 0.0586653749056332 <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> |
| results/unified_astro/qwen3_5/summary.csv | qwen3_5 | full | 1308.37 tok/s <!-- source: results/unified_astro/qwen3_5/summary.csv line 2 --> | 1809.78 MB <!-- source: results/unified_astro/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_astro/qwen3_5/summary.csv line 2 --> |
| results/unified_astro/qwen3_5/summary.csv | qwen3_5 | astro | 993.53 tok/s <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> | 1878.83 MB <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> | 0.0407704870055463 <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> |
| results/unified_astro/roberta/summary.csv | roberta | full | 22842.3 tok/s <!-- source: results/unified_astro/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_astro/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_astro/roberta/summary.csv line 2 --> |
| results/unified_astro/roberta/summary.csv | roberta | astro | 22699.84 tok/s <!-- source: results/unified_astro/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_astro/roberta/summary.csv line 3 --> | 0.0295981263740079 <!-- source: results/unified_astro/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | astro | Elapsed 0.233s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 17 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 17 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 17 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | astro_quantizer | Elapsed 0.186s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 18 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 18 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 18 --> |
