# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_block_transformer_hiba/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 2 --> |
| results/unified_block_transformer_hiba/distilbert/summary.csv | distilbert | block_transformer_hiba | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 3 --> |
| results/unified_block_transformer_hiba/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 2 --> |
| results/unified_block_transformer_hiba/gpt2/summary.csv | gpt2 | block_transformer_hiba | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 3 --> |
| results/unified_block_transformer_hiba/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 2 --> |
| results/unified_block_transformer_hiba/roberta/summary.csv | roberta | block_transformer_hiba | PASS_NO_COMPRESSION | 81.42% <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 3 --> | 0.8247 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_block_transformer_hiba/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 2 --> |
| results/unified_block_transformer_hiba/distilbert/summary.csv | distilbert | block_transformer_hiba | 66955010 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 3 --> | 67544858 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 3 --> | 589848 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 3 --> |
| results/unified_block_transformer_hiba/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 2 --> |
| results/unified_block_transformer_hiba/gpt2/summary.csv | gpt2 | block_transformer_hiba | 124441344 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 3 --> | 125621040 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 3 --> | 1179696 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 3 --> |
| results/unified_block_transformer_hiba/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 2 --> |
| results/unified_block_transformer_hiba/roberta/summary.csv | roberta | block_transformer_hiba | 124647170 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 3 --> | 132913970 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 3 --> | 1179696 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_block_transformer_hiba/distilbert/summary.csv | distilbert | full | 51289.14 tok/s <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 2 --> |
| results/unified_block_transformer_hiba/distilbert/summary.csv | distilbert | block_transformer_hiba | 25262.33 tok/s <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 3 --> | 326.45 MB <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_block_transformer_hiba/distilbert/summary.csv line 3 --> |
| results/unified_block_transformer_hiba/gpt2/summary.csv | gpt2 | full | 26878.45 tok/s <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 2 --> |
| results/unified_block_transformer_hiba/gpt2/summary.csv | gpt2 | block_transformer_hiba | 12625.36 tok/s <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 3 --> | 568.95 MB <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_block_transformer_hiba/gpt2/summary.csv line 3 --> |
| results/unified_block_transformer_hiba/roberta/summary.csv | roberta | full | 28086.77 tok/s <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 2 --> |
| results/unified_block_transformer_hiba/roberta/summary.csv | roberta | block_transformer_hiba | 11047.94 tok/s <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 3 --> | 657.23 MB <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_block_transformer_hiba/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.60% | `results/rank_ablation/lora0_agnews/summary.csv` |
| F1 Score | 0.9423 | `results/rank_ablation/lora0_agnews/summary.csv` |
| Compression | 1.000x | `results/rank_ablation/lora0_agnews/summary.csv` |
| Compressed Layers | 0 | `results/rank_ablation/lora0_agnews/summary.csv` |
| Fallback Layers | 0 | `results/rank_ablation/lora0_agnews/summary.csv` |
| Timestamp | unknown | `results/rank_ablation/lora0_agnews/summary.csv` |

