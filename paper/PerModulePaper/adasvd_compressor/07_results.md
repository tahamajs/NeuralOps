# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_adasvd/*/summary.csv`:

| Model | Status | Accuracy (%) | F1 Score | Params Before | Params After | Compression Rate | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS | 90.48% <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 0.9077 <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 1.0000x <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 182.94 <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS | 73.39% <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 0.7778 <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 109,737,258 <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 1.1359x <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 610.89 <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> |
| GPT-2 | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 1.0000x <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 628.41 <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> |
