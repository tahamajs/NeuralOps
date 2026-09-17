# 7. Empirical Results

## 7.1 Unified Benchmark Results (SST-2)
From `results/unified_adaptive_span/*/summary.csv`:

| Model | Compression Status | Accuracy (%) | F1 Score | Params Before | Params After | Peak VRAM (MB) | Tokens/sec |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS_NO_COMPRESSION | 89.45% <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 0.8928 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 81,719,120 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 382.72 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> | 27,877.8 <!-- source: results/unified_adaptive_span/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 0.8124 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 161,262,494 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 688.19 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> | 13,873.9 <!-- source: results/unified_adaptive_span/roberta/summary.csv line 3 --> |
| GPT-2 | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 175,230,876 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 754.76 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> | 13,446.5 <!-- source: results/unified_adaptive_span/gpt2/summary.csv line 3 --> |
| Qwen 3.5 | PASS | 67.89% <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 0.5570 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 763,217,728 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 795,584,368 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 1918.44 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> | 5,640.7 <!-- source: results/unified_adaptive_span/qwen3_5/summary.csv line 3 --> |

## 7.2 Multi-Dataset Evaluation (DistilBERT)
- AG News: Accuracy 65.5%, Peak VRAM 613.69 MB, Throughput 110,642 tokens/sec <!-- source: results/agnews_unified_adaptive_span/distilbert/summary.csv line 2 -->
- MNLI: Accuracy 34.6%, Peak VRAM 411.9 MB, Throughput 39,268 tokens/sec <!-- source: results/mnli_unified_adaptive_span/distilbert/summary.csv line 2 -->
