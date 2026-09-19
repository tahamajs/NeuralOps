# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_mamba_inject/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 2 --> |
| results/unified_mamba_inject/distilbert/summary.csv | distilbert | mamba_inject | SKIP_BERT_ARCHITECTURAL | 0.0% <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 3 --> | 0.9125 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 3 --> |
| results/unified_mamba_inject/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 2 --> |
| results/unified_mamba_inject/gpt2/summary.csv | gpt2 | mamba_inject | SKIP_GPT_ARCHITECTURAL | 0.0% <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 3 --> | 0.301 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 3 --> |
| results/unified_mamba_inject/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_mamba_inject/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 2 --> |
| results/unified_mamba_inject/roberta/summary.csv | roberta | mamba_inject | SKIP_ROBERTA_ARCHITECTURAL | 0.0% <!-- source: results/unified_mamba_inject/roberta/summary.csv line 3 --> | 0.7277 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_mamba_inject/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 2 --> |
| results/unified_mamba_inject/distilbert/summary.csv | distilbert | mamba_inject | 66955010 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 3 --> |
| results/unified_mamba_inject/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 2 --> |
| results/unified_mamba_inject/gpt2/summary.csv | gpt2 | mamba_inject | 124441344 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 3 --> |
| results/unified_mamba_inject/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba_inject/roberta/summary.csv line 2 --> |
| results/unified_mamba_inject/roberta/summary.csv | roberta | mamba_inject | 124647170 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba_inject/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_mamba_inject/distilbert/summary.csv | distilbert | full | 146889.0 tok/s <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 2 --> |
| results/unified_mamba_inject/distilbert/summary.csv | distilbert | mamba_inject | 146889.0 tok/s <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 3 --> | 389.51 MB <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_mamba_inject/distilbert/summary.csv line 3 --> |
| results/unified_mamba_inject/gpt2/summary.csv | gpt2 | full | 66502.29 tok/s <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 2 --> |
| results/unified_mamba_inject/gpt2/summary.csv | gpt2 | mamba_inject | 66502.29 tok/s <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 3 --> | 793.19 MB <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_mamba_inject/gpt2/summary.csv line 3 --> |
| results/unified_mamba_inject/roberta/summary.csv | roberta | full | 75988.96 tok/s <!-- source: results/unified_mamba_inject/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_mamba_inject/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 2 --> |
| results/unified_mamba_inject/roberta/summary.csv | roberta | mamba_inject | 75988.96 tok/s <!-- source: results/unified_mamba_inject/roberta/summary.csv line 3 --> | 712.34 MB <!-- source: results/unified_mamba_inject/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_mamba_inject/roberta/summary.csv line 3 --> |
