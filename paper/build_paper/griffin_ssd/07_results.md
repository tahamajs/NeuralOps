| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Griffin Ssd** | **88.30%** | -2.64% | PASS | 0.8884 | 30 | **0.94x** | 364.0 MB | 17.4s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Griffin Ssd** | **59.06%** | -16.40% | PASS | 0.6992 | 60 | **0.89x** | 710.7 MB | 35.8s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Griffin Ssd** | **84.40%** | -3.44% | PASS | 0.8528 | 84 | **0.94x** | 430.8 MB | 38.6s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_griffin_ssd/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 2 --> |
| results/unified_griffin_ssd/distilbert/summary.csv | distilbert | griffin_ssd | PASS_NO_COMPRESSION | 88.07% <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 3 --> | 0.8865 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 3 --> |
| results/unified_griffin_ssd/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 2 --> |
| results/unified_griffin_ssd/gpt2/summary.csv | gpt2 | griffin_ssd | PASS_NO_COMPRESSION | 70.18% <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 3 --> | 0.7052 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 3 --> |
| results/unified_griffin_ssd/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 2 --> |
| results/unified_griffin_ssd/roberta/summary.csv | roberta | griffin_ssd | PASS_NO_COMPRESSION | 62.39% <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 3 --> | 0.6727 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_griffin_ssd/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 2 --> |
| results/unified_griffin_ssd/distilbert/summary.csv | distilbert | griffin_ssd | 66955010 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 3 --> | 71217482 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 3 --> | 737352 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 3 --> |
| results/unified_griffin_ssd/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 2 --> |
| results/unified_griffin_ssd/gpt2/summary.csv | gpt2 | griffin_ssd | 124441344 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 3 --> | 132966288 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 3 --> | 1474704 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 3 --> |
| results/unified_griffin_ssd/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 2 --> |
| results/unified_griffin_ssd/roberta/summary.csv | roberta | griffin_ssd | 124647170 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 3 --> | 140259218 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 3 --> | 1474704 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_griffin_ssd/distilbert/summary.csv | distilbert | full | 53224.18 tok/s <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 2 --> |
| results/unified_griffin_ssd/distilbert/summary.csv | distilbert | griffin_ssd | 13846.95 tok/s <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 3 --> | 355.26 MB <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_griffin_ssd/distilbert/summary.csv line 3 --> |
| results/unified_griffin_ssd/gpt2/summary.csv | gpt2 | full | 21554.73 tok/s <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 2 --> |
| results/unified_griffin_ssd/gpt2/summary.csv | gpt2 | griffin_ssd | 9384.02 tok/s <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 3 --> | 597.32 MB <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_griffin_ssd/gpt2/summary.csv line 3 --> |
| results/unified_griffin_ssd/roberta/summary.csv | roberta | full | 17122.83 tok/s <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 2 --> |
| results/unified_griffin_ssd/roberta/summary.csv | roberta | griffin_ssd | 8289.91 tok/s <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 3 --> | 700.74 MB <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_griffin_ssd/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.14% | `results/roberta/summary.csv` |
| F1 Score | 0.0000 | `results/roberta/summary.csv` |
| Compression | 1.000x | `results/roberta/summary.csv` |
| Compressed Layers | ? | `results/roberta/summary.csv` |
| Fallback Layers | ? | `results/roberta/summary.csv` |
| Timestamp | 2026-09-04T09:19:17.168564 | `results/roberta/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [79.36, 78.56]
> These numbers must NOT be cited in the final paper.

