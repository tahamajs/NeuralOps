# 7. Empirical Results

## 7.1 Unified Benchmark Results
Evaluated under `results/unified_adaptive_ranks/`:

| Backbone Model | Status | Accuracy (%) | F1 Score | Params Before | Params After | Compression Rate | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS | 88.07% <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 3 --> | 0.8807 <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 3 --> | 58,107,650 <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 3 --> | 1.1523x <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 3 --> | 382.72 <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS | 77.87% <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 3 --> | 0.7749 <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 3 --> | 106,952,450 <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 3 --> | 1.1654x <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 3 --> | 679.52 <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 3 --> |
| GPT-2 | PASS | 77.52% <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 3 --> | 0.7752 <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 3 --> | 106,746,624 <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 3 --> | 1.1658x <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 3 --> | 754.76 <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 3 --> |
| Qwen 3.5 | PASS | 67.55% <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 3 --> | 0.5501 <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 3 --> | 763,217,728 <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 3 --> | 763,217,728 <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 3 --> | 1810.59 <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 3 --> |

## 7.2 Matrix Benchmark Latency
From `results/matrix_test_summary.csv`:
- DistilBERT: Latency 22.062 ms, VRAM 711.6 MB <!-- source: results/matrix_test_summary.csv line 4 -->
- RoBERTa: Latency 44.021 ms, VRAM 1445.8 MB <!-- source: results/matrix_test_summary.csv line 3 -->
- GPT-2: Latency 38.915 ms, VRAM 1667.2 MB <!-- source: results/matrix_test_summary.csv line 2 -->
- Qwen 3.5: Latency 210.45 ms, VRAM 4982.1 MB <!-- source: results/matrix_test_summary.csv line 1 -->
