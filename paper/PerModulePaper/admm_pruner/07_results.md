# 7. Empirical Results

## 7.1 Unified Benchmark Results
From `results/unified_admm_pruner/*/summary.csv`:

| Model | Status | Accuracy (%) | F1 Score | Pruned Layers | Sparsity | Avg Recon Error | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS | 84.29% <!-- source: results/unified_admm_pruner/distilbert/summary.csv line 3 --> | 0.8603 <!-- source: results/unified_admm_pruner/distilbert/summary.csv line 3 --> | 36 <!-- source: results/unified_admm_pruner/distilbert/summary.csv line 3 --> | 50% <!-- source: results/unified_admm_pruner_4models_summary.csv line 1 --> | 0.3571 <!-- source: results/unified_admm_pruner/distilbert/summary.csv line 3 --> | 551.24 <!-- source: results/unified_admm_pruner/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS | 80.62% <!-- source: results/unified_admm_pruner/roberta/summary.csv line 3 --> | 0.7966 <!-- source: results/unified_admm_pruner/roberta/summary.csv line 3 --> | 72 <!-- source: results/unified_admm_pruner/roberta/summary.csv line 3 --> | 50% <!-- source: results/unified_admm_pruner_4models_summary.csv line 2 --> | 0.3513 <!-- source: results/unified_admm_pruner/roberta/summary.csv line 3 --> | 336.39 <!-- source: results/unified_admm_pruner/roberta/summary.csv line 3 --> |
| GPT-2 | PASS_LOW | 51.49% <!-- source: results/unified_admm_pruner/gpt2/summary.csv line 3 --> | 0.1278 <!-- source: results/unified_admm_pruner/gpt2/summary.csv line 3 --> | 48 <!-- source: results/unified_admm_pruner/gpt2/summary.csv line 3 --> | 50% <!-- source: results/unified_admm_pruner_4models_summary.csv line 3 --> | 0.3565 <!-- source: results/unified_admm_pruner/gpt2/summary.csv line 3 --> | 628.41 <!-- source: results/unified_admm_pruner/gpt2/summary.csv line 3 --> |
| Qwen 3.5 | PASS | 78.21% <!-- source: results/unified_admm_pruner/qwen3_5/summary.csv line 3 --> | 0.8201 <!-- source: results/unified_admm_pruner/qwen3_5/summary.csv line 3 --> | 48 <!-- source: results/unified_admm_pruner/qwen3_5/summary.csv line 3 --> | 50% | 0.3569 <!-- source: results/unified_admm_pruner/qwen3_5/summary.csv line 3 --> | 1879.76 <!-- source: results/unified_admm_pruner/qwen3_5/summary.csv line 3 --> |
