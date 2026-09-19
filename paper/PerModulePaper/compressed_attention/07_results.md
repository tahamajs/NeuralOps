# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_compressed_attention/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 2 --> |
| results/unified_compressed_attention/distilbert/summary.csv | distilbert | compressed_attention | PASS | 74.54% <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 3 --> | 0.7711 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 3 --> |
| results/unified_compressed_attention/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 2 --> |
| results/unified_compressed_attention/gpt2/summary.csv | gpt2 | compressed_attention | PASS_LOW | 50.92% <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 3 --> |
| results/unified_compressed_attention/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_compressed_attention/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 2 --> |
| results/unified_compressed_attention/roberta/summary.csv | roberta | compressed_attention | PASS_LOW | 50.92% <!-- source: results/unified_compressed_attention/roberta/summary.csv line 3 --> | 0.6743 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_compressed_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 2 --> |
| results/unified_compressed_attention/distilbert/summary.csv | distilbert | compressed_attention | 66955010 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 3 --> | 54126728 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 3 --> | 1345926 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 3 --> | 1.237x <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 3 --> |
| results/unified_compressed_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 2 --> |
| results/unified_compressed_attention/gpt2/summary.csv | gpt2 | compressed_attention | 124441344 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 3 --> | 98784780 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 3 --> | 2691852 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 3 --> | 1.2597x <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 3 --> |
| results/unified_compressed_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_compressed_attention/roberta/summary.csv line 2 --> |
| results/unified_compressed_attention/roberta/summary.csv | roberta | compressed_attention | 124647170 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 3 --> | 106077710 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 3 --> | 2691852 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 3 --> | 1.1751x <!-- source: results/unified_compressed_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_compressed_attention/distilbert/summary.csv | distilbert | full | 50722.03 tok/s <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 2 --> |
| results/unified_compressed_attention/distilbert/summary.csv | distilbert | compressed_attention | 30316.81 tok/s <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 3 --> | 283.86 MB <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_compressed_attention/distilbert/summary.csv line 3 --> |
| results/unified_compressed_attention/gpt2/summary.csv | gpt2 | full | 27133.65 tok/s <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 2 --> |
| results/unified_compressed_attention/gpt2/summary.csv | gpt2 | compressed_attention | 14495.59 tok/s <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 3 --> | 470.52 MB <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_compressed_attention/gpt2/summary.csv line 3 --> |
| results/unified_compressed_attention/roberta/summary.csv | roberta | full | 27359.13 tok/s <!-- source: results/unified_compressed_attention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_compressed_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 2 --> |
| results/unified_compressed_attention/roberta/summary.csv | roberta | compressed_attention | 14513.82 tok/s <!-- source: results/unified_compressed_attention/roberta/summary.csv line 3 --> | 567.8 MB <!-- source: results/unified_compressed_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_compressed_attention/roberta/summary.csv line 3 --> |
