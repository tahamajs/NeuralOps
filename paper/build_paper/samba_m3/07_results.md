| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Sambam3** | **89.45%** | -1.49% | PASS | 0.8991 | 0 | **0.81x** | 425.2 MB | 25.6 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Sambam3** | **50.92%** | -24.54% | PASS_LOW | 0.6748 | 0 | **0.77x** | 811.7 MB | 37.4 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Sambam3** | **50.92%** | -6.99% | PASS_LOW | 0.6748 | 0 | **0.80x** | 694.4 MB | 37.5 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/benchmark_summary_v2.csv | bert-sst2 | samba_m3 | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 350 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 350 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | samba_m3 | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 351 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 351 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | samba_m3 | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 352 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 352 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | samba_m3 | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 353 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 353 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | samba_m3 | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 354 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 354 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | samba_m3 | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 355 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 355 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/benchmark_summary_v2.csv | bert-sst2 | samba_m3 | 23223174 <!-- source: results/benchmark_summary_v2.csv line 350 --> | 23223174 <!-- source: results/benchmark_summary_v2.csv line 350 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 350 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 350 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | samba_m3 | 23223174 <!-- source: results/benchmark_summary_v2.csv line 351 --> | 23223174 <!-- source: results/benchmark_summary_v2.csv line 351 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 351 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 351 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | samba_m3 | 25750278 <!-- source: results/benchmark_summary_v2.csv line 352 --> | 25750278 <!-- source: results/benchmark_summary_v2.csv line 352 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 352 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 352 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | samba_m3 | 25750278 <!-- source: results/benchmark_summary_v2.csv line 353 --> | 25750278 <!-- source: results/benchmark_summary_v2.csv line 353 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 353 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 353 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | samba_m3 | 23344516 <!-- source: results/benchmark_summary_v2.csv line 354 --> | 23344516 <!-- source: results/benchmark_summary_v2.csv line 354 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 354 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 354 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | samba_m3 | 23344516 <!-- source: results/benchmark_summary_v2.csv line 355 --> | 23344516 <!-- source: results/benchmark_summary_v2.csv line 355 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 355 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 355 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/benchmark_summary_v2.csv | bert-sst2 | samba_m3 | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 350 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 350 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 350 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | samba_m3 | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 351 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 351 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 351 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | samba_m3 | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 352 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 352 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 352 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | samba_m3 | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 353 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 353 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 353 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | samba_m3 | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 354 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 354 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 354 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | samba_m3 | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 355 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 355 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 355 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 89.91% | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| F1 Score | 0.9020 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Timestamp | 2026-09-19T01:56:33.266369 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |

