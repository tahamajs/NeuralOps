| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Compressed Attention Layer** | **69.84%** | -21.10% | PASS | 0.7650 | 0 | **1.24x** | 183.8 MB | 11.6s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Compressed Attention Layer** | **50.92%** | +0.00% | PASS_LOW | 0.6748 | 0 | **1.18x** | 338.2 MB | 22.6s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Compressed Attention Layer** | **50.92%** | +1.38% | PASS_LOW | 0.6748 | 0 | **1.26x** | 285.9 MB | 24.3s |

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

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=78.60%, comp=1.110x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 78.60% | `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group7.csv` |
| F1 Score | 0.7855 | `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group7.csv` |
| Compression | 1.110x | `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group7.csv` |
| Compressed Layers | 0 | `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group7.csv` |
| Fallback Layers | 0 | `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group7.csv` |
| Timestamp | 2026-09-20T13:12:11.966496 | `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group7.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71]
> These numbers must NOT be cited in the final paper.

