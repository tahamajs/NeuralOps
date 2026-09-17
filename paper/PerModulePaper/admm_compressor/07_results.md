# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_admm/*/summary.csv`:

| Model | Status | Accuracy (%) | F1 Score | Params | Peak VRAM (MB) | Tokens/sec |
|---|---|---|---|---|---|---|
| DistilBERT | PASS | 90.37% <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 0.9050 <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 183.07 <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 50,559.5 <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS | 62.04% <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 0.6505 <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 520.18 <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 21,347.7 <!-- source: results/unified_admm/roberta/summary.csv line 3 --> |
| GPT-2 | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 628.41 <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 19,301.5 <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> |
| Qwen 3.5 | PASS | 88.42% <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 0.8947 <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 763,217,728 <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 1851.31 <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 1,099.7 <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> |
