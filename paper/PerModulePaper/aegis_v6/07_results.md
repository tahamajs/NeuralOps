# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_aegis_v6/*/summary.csv`:

| Model | Status | Accuracy (%) | F1 Score | Params Before | Params After | Compression Rate | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS | 87.04% <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 0.8791 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 53,449,064 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 1.2527x <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> | 301.26 <!-- source: results/unified_aegis_v6/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS | 59.98% <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 0.6161 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 104,722,382 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 1.1903x <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> | 596.26 <!-- source: results/unified_aegis_v6/roberta/summary.csv line 3 --> |
| GPT-2 | PASS_LOW | 51.61% <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 0.6713 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 97,429,452 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 1.2772x <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> | 493.70 <!-- source: results/unified_aegis_v6/gpt2/summary.csv line 3 --> |
