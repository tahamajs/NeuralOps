# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_soa_attn/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_soa_attn/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 2 --> |
| results/unified_soa_attn/distilbert/summary.csv | distilbert | soa_attn | PASS | 83.49% <!-- source: results/unified_soa_attn/distilbert/summary.csv line 3 --> | 0.8389 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 3 --> |
| results/unified_soa_attn/gpt2/summary.csv | gpt2 | full | PASS | 87.84% <!-- source: results/unified_soa_attn/gpt2/summary.csv line 2 --> | 0.8776 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 2 --> |
| results/unified_soa_attn/gpt2/summary.csv | gpt2 | soa_attn | PASS_LOW | 50.92% <!-- source: results/unified_soa_attn/gpt2/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 3 --> |
| results/unified_soa_attn/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_soa_attn/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_soa_attn/roberta/summary.csv line 2 --> |
| results/unified_soa_attn/roberta/summary.csv | roberta | soa_attn | FAIL | 0.0% <!-- source: results/unified_soa_attn/roberta/summary.csv line 3 --> | 0.7277 <!-- source: results/unified_soa_attn/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_soa_attn/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soa_attn/distilbert/summary.csv line 2 --> |
| results/unified_soa_attn/distilbert/summary.csv | distilbert | soa_attn | 66955010 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 3 --> | 57370376 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 3 --> | 147462 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 3 --> | 1.1671x <!-- source: results/unified_soa_attn/distilbert/summary.csv line 3 --> |
| results/unified_soa_attn/gpt2/summary.csv | gpt2 | full | 126800640 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 2 --> | 126800640 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soa_attn/gpt2/summary.csv line 2 --> |
| results/unified_soa_attn/gpt2/summary.csv | gpt2 | soa_attn | 126800640 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 3 --> | 106746636 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 3 --> | 1769484 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 3 --> | 1.1879x <!-- source: results/unified_soa_attn/gpt2/summary.csv line 3 --> |
| results/unified_soa_attn/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_soa_attn/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_soa_attn/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_soa_attn/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soa_attn/roberta/summary.csv line 2 --> |
| results/unified_soa_attn/roberta/summary.csv | roberta | soa_attn | 124647170 <!-- source: results/unified_soa_attn/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_soa_attn/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_soa_attn/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_soa_attn/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_soa_attn/distilbert/summary.csv | distilbert | full | 41917.05 tok/s <!-- source: results/unified_soa_attn/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_soa_attn/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 2 --> |
| results/unified_soa_attn/distilbert/summary.csv | distilbert | soa_attn | 31012.9 tok/s <!-- source: results/unified_soa_attn/distilbert/summary.csv line 3 --> | 279.93 MB <!-- source: results/unified_soa_attn/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soa_attn/distilbert/summary.csv line 3 --> |
| results/unified_soa_attn/gpt2/summary.csv | gpt2 | full | 9411.5 tok/s <!-- source: results/unified_soa_attn/gpt2/summary.csv line 2 --> | 391.24 MB <!-- source: results/unified_soa_attn/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 2 --> |
| results/unified_soa_attn/gpt2/summary.csv | gpt2 | soa_attn | 12419.29 tok/s <!-- source: results/unified_soa_attn/gpt2/summary.csv line 3 --> | 301.23 MB <!-- source: results/unified_soa_attn/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soa_attn/gpt2/summary.csv line 3 --> |
| results/unified_soa_attn/roberta/summary.csv | roberta | full | 26620.13 tok/s <!-- source: results/unified_soa_attn/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_soa_attn/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soa_attn/roberta/summary.csv line 2 --> |
| results/unified_soa_attn/roberta/summary.csv | roberta | soa_attn | 26620.13 tok/s <!-- source: results/unified_soa_attn/roberta/summary.csv line 3 --> | 712.34 MB <!-- source: results/unified_soa_attn/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soa_attn/roberta/summary.csv line 3 --> |
