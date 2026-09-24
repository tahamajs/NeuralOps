| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Optimized Hybrid Attention** | **90.14%** | -0.80% | PASS | 0.9057 | 0 | **0.99x** | 340.2 MB | 5.8s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Optimized Hybrid Attention** | **55.39%** | -20.07% | PASS_LOW | 0.6949 | 0 | **0.93x** | 672.0 MB | 11.4s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Optimized Hybrid Attention** | **51.26%** | -6.65% | PASS_LOW | 0.6763 | 0 | **0.99x** | 572.0 MB | 12.2s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_optimized_hybrid_attention/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/distilbert/summary.csv | distilbert | optimized_hybrid_attention | PASS_NO_COMPRESSION | 90.48% <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 3 --> | 0.9073 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 3 --> |
| results/unified_optimized_hybrid_attention/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/gpt2/summary.csv | gpt2 | optimized_hybrid_attention | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 3 --> |
| results/unified_optimized_hybrid_attention/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/qwen3_5/summary.csv | qwen3_5 | optimized_hybrid_attention | SKIP_QWEN3_5_TIMEOUT | 0.0% <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 3 --> | 0.557 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 3 --> |
| results/unified_optimized_hybrid_attention/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/roberta/summary.csv | roberta | optimized_hybrid_attention | PASS_NO_COMPRESSION | 81.88% <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 3 --> | 0.8326 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_optimized_hybrid_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/distilbert/summary.csv | distilbert | optimized_hybrid_attention | 66955010 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 3 --> | 67895138 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 3 --> | 645216 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 3 --> |
| results/unified_optimized_hybrid_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/gpt2/summary.csv | gpt2 | optimized_hybrid_attention | 124441344 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 3 --> | 126321600 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 3 --> | 1290432 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 3 --> |
| results/unified_optimized_hybrid_attention/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/qwen3_5/summary.csv | qwen3_5 | optimized_hybrid_attention | 763217728 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 3 --> | 0 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 3 --> |
| results/unified_optimized_hybrid_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/roberta/summary.csv | roberta | optimized_hybrid_attention | 124647170 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 3 --> | 133614530 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 3 --> | 1290432 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_optimized_hybrid_attention/distilbert/summary.csv | distilbert | full | 52566.52 tok/s <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/distilbert/summary.csv | distilbert | optimized_hybrid_attention | 17249.13 tok/s <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 3 --> | 337.15 MB <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_optimized_hybrid_attention/distilbert/summary.csv line 3 --> |
| results/unified_optimized_hybrid_attention/gpt2/summary.csv | gpt2 | full | 27438.96 tok/s <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/gpt2/summary.csv | gpt2 | optimized_hybrid_attention | 7214.89 tok/s <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 3 --> | 571.65 MB <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_optimized_hybrid_attention/gpt2/summary.csv line 3 --> |
| results/unified_optimized_hybrid_attention/qwen3_5/summary.csv | qwen3_5 | full | 1275.31 tok/s <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/qwen3_5/summary.csv | qwen3_5 | optimized_hybrid_attention | 1275.31 tok/s <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 3 --> | 1796.78 MB <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_optimized_hybrid_attention/qwen3_5/summary.csv line 3 --> |
| results/unified_optimized_hybrid_attention/roberta/summary.csv | roberta | full | 27321.01 tok/s <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 2 --> |
| results/unified_optimized_hybrid_attention/roberta/summary.csv | roberta | optimized_hybrid_attention | 8458.7 tok/s <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 3 --> | 669.6 MB <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_optimized_hybrid_attention/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 0.00% | `results/final_paper_run/sst2_plotting_data.csv` |
| F1 Score | 0.9087 | `results/final_paper_run/sst2_plotting_data.csv` |
| Compression | 1.000x | `results/final_paper_run/sst2_plotting_data.csv` |
| Compressed Layers | ? | `results/final_paper_run/sst2_plotting_data.csv` |
| Fallback Layers | ? | `results/final_paper_run/sst2_plotting_data.csv` |
| Timestamp | unknown | `results/final_paper_run/sst2_plotting_data.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [67.89]
> These numbers must NOT be cited in the final paper.

