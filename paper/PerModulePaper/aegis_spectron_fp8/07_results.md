# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_aegis_spectron_fp8/*/summary.csv`:

| Model | Status | Accuracy (%) | Params Before | Params After | Compression Rate | Peak VRAM (MB) | Tokens/sec |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS | 79.59% <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 54,753,068 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 1.2229x <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 290.29 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 9,858.7 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS_LOW | 49.43% <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 107,330,390 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 1.1613x <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 572.68 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 4,840.6 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> |
| GPT-2 | PASS | 58.83% <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 100,037,460 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 1.2439x <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 643.49 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 5,217.5 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> |
