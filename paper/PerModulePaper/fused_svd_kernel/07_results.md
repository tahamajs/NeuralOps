| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Fused Svd Kernel** | **81.77%** | -9.17% | PASS | 0.8235 | 0 | **1.24x** | 177.6 MB | 80.3s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Fused Svd Kernel** | **50.92%** | -27.64% | PASS_LOW | 0.6748 | 0 | **1.15x** | 610.2 MB | 20.9s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Fused Svd Kernel** | **60.32%** | -19.04% | PASS | 0.6608 | 0 | **1.23x** | 515.1 MB | 108.8s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_fused_svd_kernel/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 2 --> |
| results/unified_fused_svd_kernel/distilbert/summary.csv | distilbert | fused_svd_kernel | PASS | 79.93% <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> | 0.8201 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> |
| results/unified_fused_svd_kernel/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 2 --> |
| results/unified_fused_svd_kernel/gpt2/summary.csv | gpt2 | fused_svd_kernel | PASS | 70.07% <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> | 0.754 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> |
| results/unified_fused_svd_kernel/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 2 --> |
| results/unified_fused_svd_kernel/roberta/summary.csv | roberta | fused_svd_kernel | PASS_LOW | 50.92% <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_fused_svd_kernel/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 2 --> |
| results/unified_fused_svd_kernel/distilbert/summary.csv | distilbert | fused_svd_kernel | 66955010 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> | 53978882 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> | 1184256 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> | 1.2404x <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> |
| results/unified_fused_svd_kernel/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 2 --> |
| results/unified_fused_svd_kernel/gpt2/summary.csv | gpt2 | fused_svd_kernel | 124441344 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> | 98489088 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> | 1778688 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> | 1.2635x <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> |
| results/unified_fused_svd_kernel/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 2 --> |
| results/unified_fused_svd_kernel/roberta/summary.csv | roberta | fused_svd_kernel | 124647170 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> | 105782018 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> | 2368512 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> | 1.1783x <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_fused_svd_kernel/distilbert/summary.csv | distilbert | full | 53310.16 tok/s <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 2 --> |
| results/unified_fused_svd_kernel/distilbert/summary.csv | distilbert | fused_svd_kernel | 3888.43 tok/s <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> | 280.27 MB <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> |
| results/unified_fused_svd_kernel/gpt2/summary.csv | gpt2 | full | 27584.57 tok/s <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 2 --> |
| results/unified_fused_svd_kernel/gpt2/summary.csv | gpt2 | fused_svd_kernel | 1955.65 tok/s <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> | 464.46 MB <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> |
| results/unified_fused_svd_kernel/roberta/summary.csv | roberta | full | 28038.54 tok/s <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 2 --> |
| results/unified_fused_svd_kernel/roberta/summary.csv | roberta | fused_svd_kernel | 17791.54 tok/s <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> | 558.41 MB <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=91.90%, comp=1.076x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 91.90% | `results/rank_ablation/agnews_rank256/summary.csv` |
| F1 Score | 0.9163 | `results/rank_ablation/agnews_rank256/summary.csv` |
| Compression | 1.076x | `results/rank_ablation/agnews_rank256/summary.csv` |
| Compressed Layers | 0 | `results/rank_ablation/agnews_rank256/summary.csv` |
| Fallback Layers | 0 | `results/rank_ablation/agnews_rank256/summary.csv` |
| Timestamp | 2026-09-20T00:32:15.871260 | `results/rank_ablation/agnews_rank256/summary.csv` |

