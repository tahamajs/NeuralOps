# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_aegis_v2/*/summary.csv`:

| Model | Status | Accuracy (%) | Params Before | Params After | Compression Rate | Peak VRAM (MB) | Tokens/sec |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS | 72.36% <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 55,523,882 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 1.2059x <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 195.70 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> | 14,953.2 <!-- source: results/unified_aegis_v2/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS | 62.39% <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 108,872,018 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 1.1449x <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 591.16 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> | 7,112.6 <!-- source: results/unified_aegis_v2/roberta/summary.csv line 3 --> |
| GPT-2 | PASS_LOW | 52.98% <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 101,579,088 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 1.2251x <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 476.00 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> | 8,738.5 <!-- source: results/unified_aegis_v2/gpt2/summary.csv line 3 --> |
