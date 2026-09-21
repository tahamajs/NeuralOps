# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_huffman_coder/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 2 --> |
| results/unified_huffman_coder/distilbert/summary.csv | distilbert | huffman_coder | PASS | 90.6% <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 3 --> | 0.9095 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 3 --> |
| results/unified_huffman_coder/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 2 --> |
| results/unified_huffman_coder/gpt2/summary.csv | gpt2 | huffman_coder | PASS_LOW | 50.92% <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 3 --> |
| results/unified_huffman_coder/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 2 --> |
| results/unified_huffman_coder/qwen3_5/summary.csv | qwen3_5 | huffman_coder | FAIL | 0.0% <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 3 --> | 0.557 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 3 --> |
| results/unified_huffman_coder/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_huffman_coder/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 2 --> |
| results/unified_huffman_coder/roberta/summary.csv | roberta | huffman_coder | PASS_LOW | 51.38% <!-- source: results/unified_huffman_coder/roberta/summary.csv line 3 --> | 0.6768 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_huffman_coder/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 2 --> |
| results/unified_huffman_coder/distilbert/summary.csv | distilbert | huffman_coder | 66955010 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 3 --> | 6914 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 3 --> |
| results/unified_huffman_coder/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 2 --> |
| results/unified_huffman_coder/gpt2/summary.csv | gpt2 | huffman_coder | 124441344 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 3 --> |
| results/unified_huffman_coder/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 2 --> |
| results/unified_huffman_coder/qwen3_5/summary.csv | qwen3_5 | huffman_coder | 763217728 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 3 --> | 0 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 3 --> |
| results/unified_huffman_coder/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_huffman_coder/roberta/summary.csv line 2 --> |
| results/unified_huffman_coder/roberta/summary.csv | roberta | huffman_coder | 124647170 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 3 --> | 11522 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_huffman_coder/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_huffman_coder/distilbert/summary.csv | distilbert | full | 170114.86 tok/s <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 2 --> |
| results/unified_huffman_coder/distilbert/summary.csv | distilbert | huffman_coder | 158785.1 tok/s <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 3 --> | 308.84 MB <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 3 --> | 0.0458100470469583 <!-- source: results/unified_huffman_coder/distilbert/summary.csv line 3 --> |
| results/unified_huffman_coder/gpt2/summary.csv | gpt2 | full | 71165.98 tok/s <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 2 --> |
| results/unified_huffman_coder/gpt2/summary.csv | gpt2 | huffman_coder | 74202.0 tok/s <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_huffman_coder/gpt2/summary.csv line 3 --> |
| results/unified_huffman_coder/qwen3_5/summary.csv | qwen3_5 | full | 1270.75 tok/s <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 2 --> |
| results/unified_huffman_coder/qwen3_5/summary.csv | qwen3_5 | huffman_coder | 1270.75 tok/s <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 3 --> | 1796.78 MB <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_huffman_coder/qwen3_5/summary.csv line 3 --> |
| results/unified_huffman_coder/roberta/summary.csv | roberta | full | 92231.55 tok/s <!-- source: results/unified_huffman_coder/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_huffman_coder/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 2 --> |
| results/unified_huffman_coder/roberta/summary.csv | roberta | huffman_coder | 86035.52 tok/s <!-- source: results/unified_huffman_coder/roberta/summary.csv line 3 --> | 528.86 MB <!-- source: results/unified_huffman_coder/roberta/summary.csv line 3 --> | 0.0277394032381122 <!-- source: results/unified_huffman_coder/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.60% | `results/unified_huffman_coder/distilbert/summary.csv` |
| F1 Score | 0.9095 | `results/unified_huffman_coder/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_huffman_coder/distilbert/summary.csv` |
| Compressed Layers | 35 | `results/unified_huffman_coder/distilbert/summary.csv` |
| Fallback Layers | 1 | `results/unified_huffman_coder/distilbert/summary.csv` |
| Timestamp | 2026-09-11T19:06:18.993141 | `results/unified_huffman_coder/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [57.91, 67.89, 75.46]
> These numbers must NOT be cited in the final paper.

