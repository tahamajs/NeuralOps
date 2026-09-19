# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | adaptive_compressor | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | adaptive_compressor | PASS | 76.95% <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 0.8016 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | adaptive_compressor | PASS_NO_COMPRESSION | 85.32% <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 0.8447 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | adaptive_compressor | 66955010 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | adaptive_compressor | 124441344 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 124177153 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 326401 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 1.0021x <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | full | 127325954 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 127325954 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | adaptive_compressor | 127325954 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 127325954 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 1474560 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | full | 50739.73 tok/s <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_compressor/distilbert/summary.csv | distilbert | adaptive_compressor | 49536.87 tok/s <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 183.07 MB <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | full | 23311.22 tok/s <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_compressor/gpt2/summary.csv | gpt2 | adaptive_compressor | 22313.99 tok/s <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 638.52 MB <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 0.1982078150637553 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | full | 25000.16 tok/s <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 299.92 MB <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 --> |
| results/unified_adaptive_compressor/roberta/summary.csv | roberta | adaptive_compressor | 11130.61 tok/s <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 326.48 MB <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> |
