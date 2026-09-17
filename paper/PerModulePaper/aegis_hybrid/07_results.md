# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_aegis_hybrid/*/summary.csv`:

| Model | Status | Accuracy (%) | Params Before | Params After | Compression Rate | Peak VRAM (MB) | Tokens/sec |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS | 84.98% <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 59,652,626 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 1.1224x <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 319.22 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> | 3,803.0 <!-- source: results/unified_aegis_hybrid/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS_LOW | 51.26% <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 108,872,018 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 1.1449x <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 596.29 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> | 7,060.8 <!-- source: results/unified_aegis_hybrid/roberta/summary.csv line 3 --> |
| GPT-2 | PASS | 59.52% <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 101,579,088 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 1.2251x <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 476.00 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> | 8,474.5 <!-- source: results/unified_aegis_hybrid/gpt2/summary.csv line 3 --> |

## 7.2 Matrix Benchmark Latency
From `results/matrix_test_summary.csv`:
- DistilBERT: Latency 3.532 ms, Peak VRAM 809.0 MB <!-- source: results/matrix_test_summary.csv line 4 -->
- RoBERTa: Latency 4.520 ms, Peak VRAM 1587.8 MB <!-- source: results/matrix_test_summary.csv line 3 -->
- GPT-2: Latency 5.286 ms, Peak VRAM 1442.0 MB <!-- source: results/matrix_test_summary.csv line 2 -->
- Qwen 3.5: Latency 14.619 ms, Peak VRAM 4255.0 MB <!-- source: results/matrix_test_summary.csv line 1 -->
