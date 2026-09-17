# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_aegis_v1/*/summary.csv`:

| Model | Status | Accuracy (%) | F1 Score | Params Before | Params After | Peak VRAM (MB) | Tokens/sec |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS_NO_COMPRESSION | 61.81% <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 0.7209 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 67,741,538 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 329.48 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 12,321.9 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS_NO_COMPRESSION | 79.13% <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 0.8131 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 133,307,330 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 674.79 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 6,386.4 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> |
| GPT-2 | PASS_NO_COMPRESSION | 70.99% <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 0.7668 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 126,014,400 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 589.37 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 9,887.9 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> |

## 7.2 Matrix Benchmark Latency
From `results/matrix_test_summary.csv`:
- DistilBERT: Latency 43.797 ms, Peak VRAM 804.7 MB <!-- source: results/matrix_test_summary.csv line 4 -->
- RoBERTa: Latency 56.309 ms, Peak VRAM 1579.2 MB <!-- source: results/matrix_test_summary.csv line 3 -->
- GPT-2: Latency 58.335 ms, Peak VRAM 1440.7 MB <!-- source: results/matrix_test_summary.csv line 2 -->
- Qwen 3.5: Latency 45.451 ms, Peak VRAM 5072.9 MB <!-- source: results/matrix_test_summary.csv line 1 -->
