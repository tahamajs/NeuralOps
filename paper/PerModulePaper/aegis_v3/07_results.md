# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_aegis_v3/*/summary.csv`:

| Model | Status | Accuracy (%) | Params Before | Params After | Peak VRAM (MB) | Tokens/sec |
|---|---|---|---|---|---|---|
| DistilBERT | PASS_NO_COMPRESSION | 78.44% <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 67,741,538 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 208.28 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 12,530.6 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS_NO_COMPRESSION | 55.16% <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 133,307,330 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 688.09 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 5,952.9 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> |
| GPT-2 | PASS_NO_COMPRESSION | 54.82% <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 126,014,400 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 594.88 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 8,863.4 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> |
