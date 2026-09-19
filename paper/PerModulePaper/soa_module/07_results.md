# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_soa_module/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_soa_module/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_soa_module/distilbert/summary.csv line 2 --> |
| results/unified_soa_module/distilbert/summary.csv | distilbert | soa_module | PASS | 89.22% <!-- source: results/unified_soa_module/distilbert/summary.csv line 3 --> | 0.898 <!-- source: results/unified_soa_module/distilbert/summary.csv line 3 --> |
| results/unified_soa_module/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_soa_module/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_soa_module/gpt2/summary.csv line 2 --> |
| results/unified_soa_module/gpt2/summary.csv | gpt2 | soa_module | PASS | 66.74% <!-- source: results/unified_soa_module/gpt2/summary.csv line 3 --> | 0.7456 <!-- source: results/unified_soa_module/gpt2/summary.csv line 3 --> |
| results/unified_soa_module/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_soa_module/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_soa_module/roberta/summary.csv line 2 --> |
| results/unified_soa_module/roberta/summary.csv | roberta | soa_module | PASS | 59.75% <!-- source: results/unified_soa_module/roberta/summary.csv line 3 --> | 0.7018 <!-- source: results/unified_soa_module/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_soa_module/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_soa_module/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_soa_module/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_soa_module/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soa_module/distilbert/summary.csv line 2 --> |
| results/unified_soa_module/distilbert/summary.csv | distilbert | soa_module | 66955010 <!-- source: results/unified_soa_module/distilbert/summary.csv line 3 --> | 60282626 <!-- source: results/unified_soa_module/distilbert/summary.csv line 3 --> | 405504 <!-- source: results/unified_soa_module/distilbert/summary.csv line 3 --> | 1.1107x <!-- source: results/unified_soa_module/distilbert/summary.csv line 3 --> |
| results/unified_soa_module/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_soa_module/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_soa_module/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_soa_module/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soa_module/gpt2/summary.csv line 2 --> |
| results/unified_soa_module/gpt2/summary.csv | gpt2 | soa_module | 124441344 <!-- source: results/unified_soa_module/gpt2/summary.csv line 3 --> | 111096576 <!-- source: results/unified_soa_module/gpt2/summary.csv line 3 --> | 811008 <!-- source: results/unified_soa_module/gpt2/summary.csv line 3 --> | 1.1201x <!-- source: results/unified_soa_module/gpt2/summary.csv line 3 --> |
| results/unified_soa_module/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_soa_module/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_soa_module/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_soa_module/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soa_module/roberta/summary.csv line 2 --> |
| results/unified_soa_module/roberta/summary.csv | roberta | soa_module | 124647170 <!-- source: results/unified_soa_module/roberta/summary.csv line 3 --> | 118389506 <!-- source: results/unified_soa_module/roberta/summary.csv line 3 --> | 811008 <!-- source: results/unified_soa_module/roberta/summary.csv line 3 --> | 1.0529x <!-- source: results/unified_soa_module/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_soa_module/distilbert/summary.csv | distilbert | full | 27554.18 tok/s <!-- source: results/unified_soa_module/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_soa_module/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soa_module/distilbert/summary.csv line 2 --> |
| results/unified_soa_module/distilbert/summary.csv | distilbert | soa_module | 13809.54 tok/s <!-- source: results/unified_soa_module/distilbert/summary.csv line 3 --> | 293.12 MB <!-- source: results/unified_soa_module/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soa_module/distilbert/summary.csv line 3 --> |
| results/unified_soa_module/gpt2/summary.csv | gpt2 | full | 14789.31 tok/s <!-- source: results/unified_soa_module/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_soa_module/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soa_module/gpt2/summary.csv line 2 --> |
| results/unified_soa_module/gpt2/summary.csv | gpt2 | soa_module | 6485.02 tok/s <!-- source: results/unified_soa_module/gpt2/summary.csv line 3 --> | 509.14 MB <!-- source: results/unified_soa_module/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soa_module/gpt2/summary.csv line 3 --> |
| results/unified_soa_module/roberta/summary.csv | roberta | full | 13296.55 tok/s <!-- source: results/unified_soa_module/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_soa_module/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soa_module/roberta/summary.csv line 2 --> |
| results/unified_soa_module/roberta/summary.csv | roberta | soa_module | 7661.01 tok/s <!-- source: results/unified_soa_module/roberta/summary.csv line 3 --> | 595.15 MB <!-- source: results/unified_soa_module/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soa_module/roberta/summary.csv line 3 --> |
