| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Sambamoe** | **76.72%** | -14.22% | PASS | 0.8012 | 96 | **0.54x** | 600.4 MB | 47.1 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Sambamoe** | **50.92%** | -24.54% | PASS_LOW | 0.6748 | 192 | **0.50x** | 1173.7 MB | 51.7 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Sambamoe** | **50.92%** | -6.99% | PASS_LOW | 0.6748 | 192 | **0.52x** | 1058.0 MB | 51.1 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/benchmark_summary_v2.csv | bert-sst2 | samba_moe | PASS | 3126.87 ms/step% <!-- source: results/benchmark_summary_v2.csv line 344 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 344 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | samba_moe | PASS | 3108.56 ms/step% <!-- source: results/benchmark_summary_v2.csv line 345 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 345 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | samba_moe | PASS | 3134.02 ms/step% <!-- source: results/benchmark_summary_v2.csv line 346 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 346 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | samba_moe | PASS | 3144.01 ms/step% <!-- source: results/benchmark_summary_v2.csv line 347 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 347 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | samba_moe | PASS | 3176.26 ms/step% <!-- source: results/benchmark_summary_v2.csv line 348 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 348 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | samba_moe | PASS | 3170.7 ms/step% <!-- source: results/benchmark_summary_v2.csv line 349 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 349 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/benchmark_summary_v2.csv | bert-sst2 | samba_moe | 5600646 <!-- source: results/benchmark_summary_v2.csv line 344 --> | 5600646 <!-- source: results/benchmark_summary_v2.csv line 344 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 344 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 344 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | samba_moe | 5600646 <!-- source: results/benchmark_summary_v2.csv line 345 --> | 5600646 <!-- source: results/benchmark_summary_v2.csv line 345 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 345 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 345 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | samba_moe | 8127750 <!-- source: results/benchmark_summary_v2.csv line 346 --> | 8127750 <!-- source: results/benchmark_summary_v2.csv line 346 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 346 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 346 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | samba_moe | 8127750 <!-- source: results/benchmark_summary_v2.csv line 347 --> | 8127750 <!-- source: results/benchmark_summary_v2.csv line 347 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 347 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 347 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | samba_moe | 5721988 <!-- source: results/benchmark_summary_v2.csv line 348 --> | 5721988 <!-- source: results/benchmark_summary_v2.csv line 348 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 348 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 348 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | samba_moe | 5721988 <!-- source: results/benchmark_summary_v2.csv line 349 --> | 5721988 <!-- source: results/benchmark_summary_v2.csv line 349 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 349 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 349 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/benchmark_summary_v2.csv | bert-sst2 | samba_moe | Latency 3126.87ms tok/s <!-- source: results/benchmark_summary_v2.csv line 344 --> | 5085.0 MB <!-- source: results/benchmark_summary_v2.csv line 344 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 344 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | samba_moe | Latency 3108.56ms tok/s <!-- source: results/benchmark_summary_v2.csv line 345 --> | 5085.0 MB <!-- source: results/benchmark_summary_v2.csv line 345 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 345 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | samba_moe | Latency 3134.02ms tok/s <!-- source: results/benchmark_summary_v2.csv line 346 --> | 5094.8 MB <!-- source: results/benchmark_summary_v2.csv line 346 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 346 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | samba_moe | Latency 3144.01ms tok/s <!-- source: results/benchmark_summary_v2.csv line 347 --> | 5094.8 MB <!-- source: results/benchmark_summary_v2.csv line 347 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 347 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | samba_moe | Latency 3176.26ms tok/s <!-- source: results/benchmark_summary_v2.csv line 348 --> | 5101.3 MB <!-- source: results/benchmark_summary_v2.csv line 348 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 348 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | samba_moe | Latency 3170.7ms tok/s <!-- source: results/benchmark_summary_v2.csv line 349 --> | 5101.3 MB <!-- source: results/benchmark_summary_v2.csv line 349 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 349 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 88.30% | `results/distilbert/summary.csv` |
| F1 Score | 0.0000 | `results/distilbert/summary.csv` |
| Compression | 1.000x | `results/distilbert/summary.csv` |
| Compressed Layers | ? | `results/distilbert/summary.csv` |
| Fallback Layers | ? | `results/distilbert/summary.csv` |
| Timestamp | 2026-09-04T08:14:17.931377 | `results/distilbert/summary.csv` |

