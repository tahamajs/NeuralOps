# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_mamba/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_mamba/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_mamba/distilbert/summary.csv line 2 --> |
| results/unified_mamba/distilbert/summary.csv | distilbert | mamba | PASS_LOW | 49.08% <!-- source: results/unified_mamba/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_mamba/distilbert/summary.csv line 3 --> |
| results/unified_mamba/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_mamba/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_mamba/gpt2/summary.csv line 2 --> |
| results/unified_mamba/gpt2/summary.csv | gpt2 | mamba | PASS_LOW | 48.85% <!-- source: results/unified_mamba/gpt2/summary.csv line 3 --> | 0.231 <!-- source: results/unified_mamba/gpt2/summary.csv line 3 --> |
| results/unified_mamba/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_mamba/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_mamba/roberta/summary.csv line 2 --> |
| results/unified_mamba/roberta/summary.csv | roberta | mamba | PASS_NO_COMPRESSION | 50.92% <!-- source: results/unified_mamba/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_mamba/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_mamba/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_mamba/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_mamba/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba/distilbert/summary.csv line 2 --> |
| results/unified_mamba/distilbert/summary.csv | distilbert | mamba | 66955010 <!-- source: results/unified_mamba/distilbert/summary.csv line 3 --> | 63784724 <!-- source: results/unified_mamba/distilbert/summary.csv line 3 --> | 368640 <!-- source: results/unified_mamba/distilbert/summary.csv line 3 --> | 1.0497x <!-- source: results/unified_mamba/distilbert/summary.csv line 3 --> |
| results/unified_mamba/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_mamba/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_mamba/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba/gpt2/summary.csv line 2 --> |
| results/unified_mamba/gpt2/summary.csv | gpt2 | mamba | 124441344 <!-- source: results/unified_mamba/gpt2/summary.csv line 3 --> | 118100772 <!-- source: results/unified_mamba/gpt2/summary.csv line 3 --> | 737280 <!-- source: results/unified_mamba/gpt2/summary.csv line 3 --> | 1.0537x <!-- source: results/unified_mamba/gpt2/summary.csv line 3 --> |
| results/unified_mamba/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_mamba/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_mamba/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba/roberta/summary.csv line 2 --> |
| results/unified_mamba/roberta/summary.csv | roberta | mamba | 124647170 <!-- source: results/unified_mamba/roberta/summary.csv line 3 --> | 125393702 <!-- source: results/unified_mamba/roberta/summary.csv line 3 --> | 737280 <!-- source: results/unified_mamba/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_mamba/distilbert/summary.csv | distilbert | full | 53982.11 tok/s <!-- source: results/unified_mamba/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_mamba/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba/distilbert/summary.csv line 2 --> |
| results/unified_mamba/distilbert/summary.csv | distilbert | mamba | 21965.26 tok/s <!-- source: results/unified_mamba/distilbert/summary.csv line 3 --> | 537.07 MB <!-- source: results/unified_mamba/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_mamba/distilbert/summary.csv line 3 --> |
| results/unified_mamba/gpt2/summary.csv | gpt2 | full | 26553.36 tok/s <!-- source: results/unified_mamba/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_mamba/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba/gpt2/summary.csv line 2 --> |
| results/unified_mamba/gpt2/summary.csv | gpt2 | mamba | 15361.74 tok/s <!-- source: results/unified_mamba/gpt2/summary.csv line 3 --> | 649.68 MB <!-- source: results/unified_mamba/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_mamba/gpt2/summary.csv line 3 --> |
| results/unified_mamba/roberta/summary.csv | roberta | full | 28447.45 tok/s <!-- source: results/unified_mamba/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_mamba/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba/roberta/summary.csv line 2 --> |
| results/unified_mamba/roberta/summary.csv | roberta | mamba | 11475.19 tok/s <!-- source: results/unified_mamba/roberta/summary.csv line 3 --> | 852.09 MB <!-- source: results/unified_mamba/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_mamba/roberta/summary.csv line 3 --> |
