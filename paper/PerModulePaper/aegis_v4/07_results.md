# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_aegis_v4/*/summary.csv`:

| Model | Status | Accuracy (%) | Params Before | Params After | Peak VRAM (MB) | Tokens/sec |
|---|---|---|---|---|---|---|
| DistilBERT | PASS_NO_COMPRESSION | 85.21% <!-- source: results/unified_aegis_v4/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_aegis_v4/distilbert/summary.csv line 3 --> | 67,741,514 <!-- source: results/unified_aegis_v4/distilbert/summary.csv line 3 --> | 193.86 <!-- source: results/unified_aegis_v4/distilbert/summary.csv line 3 --> | 13,166.4 <!-- source: results/unified_aegis_v4/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS_NO_COMPRESSION | 57.80% <!-- source: results/unified_aegis_v4/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_aegis_v4/roberta/summary.csv line 3 --> | 133,307,282 <!-- source: results/unified_aegis_v4/roberta/summary.csv line 3 --> | 676.53 <!-- source: results/unified_aegis_v4/roberta/summary.csv line 3 --> | 6,149.9 <!-- source: results/unified_aegis_v4/roberta/summary.csv line 3 --> |
| GPT-2 | PASS_NO_COMPRESSION | 70.99% <!-- source: results/unified_aegis_v4/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_aegis_v4/gpt2/summary.csv line 3 --> | 126,014,352 <!-- source: results/unified_aegis_v4/gpt2/summary.csv line 3 --> | 590.29 <!-- source: results/unified_aegis_v4/gpt2/summary.csv line 3 --> | 9,810.5 <!-- source: results/unified_aegis_v4/gpt2/summary.csv line 3 --> |
