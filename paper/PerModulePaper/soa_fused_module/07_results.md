# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_soa_fused_module/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 2 --> |
| results/unified_soa_fused_module/distilbert/summary.csv | distilbert | soa_fused_module | PASS | 81.77% <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 3 --> | 0.8285 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 3 --> |
| results/unified_soa_fused_module/gpt2/summary.csv | gpt2 | full | PASS | 87.84% <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 2 --> | 0.8776 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 2 --> |
| results/unified_soa_fused_module/gpt2/summary.csv | gpt2 | soa_fused_module | PASS_LOW | 50.92% <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 3 --> |
| results/unified_soa_fused_module/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 2 --> |
| results/unified_soa_fused_module/roberta/summary.csv | roberta | soa_fused_module | PASS_LOW | 53.21% <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 3 --> | 0.6837 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_soa_fused_module/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 2 --> |
| results/unified_soa_fused_module/distilbert/summary.csv | distilbert | soa_fused_module | 66955010 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 3 --> | 57600770 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 3 --> | 377856 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 3 --> | 1.1624x <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 3 --> |
| results/unified_soa_fused_module/gpt2/summary.csv | gpt2 | full | 126800640 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 2 --> | 126800640 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 2 --> |
| results/unified_soa_fused_module/gpt2/summary.csv | gpt2 | soa_fused_module | 126800640 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 3 --> | 107207424 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 3 --> | 2230272 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 3 --> | 1.1828x <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 3 --> |
| results/unified_soa_fused_module/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 2 --> |
| results/unified_soa_fused_module/roberta/summary.csv | roberta | soa_fused_module | 124647170 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 3 --> | 113025794 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 3 --> | 755712 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 3 --> | 1.1028x <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_soa_fused_module/distilbert/summary.csv | distilbert | full | 42523.5 tok/s <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 2 --> |
| results/unified_soa_fused_module/distilbert/summary.csv | distilbert | soa_fused_module | 22209.6 tok/s <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 3 --> | 280.44 MB <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soa_fused_module/distilbert/summary.csv line 3 --> |
| results/unified_soa_fused_module/gpt2/summary.csv | gpt2 | full | 27338.75 tok/s <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 2 --> | 391.24 MB <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 2 --> |
| results/unified_soa_fused_module/gpt2/summary.csv | gpt2 | soa_fused_module | 10748.88 tok/s <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 3 --> | 296.89 MB <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soa_fused_module/gpt2/summary.csv line 3 --> |
| results/unified_soa_fused_module/roberta/summary.csv | roberta | full | 28155.5 tok/s <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 2 --> |
| results/unified_soa_fused_module/roberta/summary.csv | roberta | soa_fused_module | 11857.43 tok/s <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 3 --> | 573.39 MB <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_soa_fused_module/roberta/summary.csv line 3 --> |
