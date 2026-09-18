# 7. Empirical Results

## 7.1 Unified Benchmark Results
Evaluated under the standardized unified benchmark (`results/unified_adaptive_compressor/`):

<!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 -->
| Backbone Model | Compression Status | Accuracy (%) | F1 Score | Params Before | Params After | Compression Rate | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|
| DistilBERT | PASS | 88.07% <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 0.8807 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 66,955,010 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 58,107,650 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 1.1523x <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> | 382.72 <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 3 --> |
| RoBERTa | PASS | 77.87% <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 0.7749 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 124,647,170 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 106,952,450 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 1.1654x <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> | 679.52 <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 3 --> |
| GPT-2 | PASS | 77.52% <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 0.7752 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 124,441,344 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 106,746,624 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 1.1658x <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> | 754.76 <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 3 --> |

## 7.2 Baseline Comparison
Baseline uncompressed models:
- DistilBERT Full: 90.71% accuracy, 66.95M parameters <!-- source: results/unified_adaptive_compressor/distilbert/summary.csv line 2 -->
- RoBERTa Full: 78.56% accuracy, 124.65M parameters <!-- source: results/unified_adaptive_compressor/roberta/summary.csv line 2 -->
- GPT-2 Full: 79.36% accuracy, 124.44M parameters <!-- source: results/unified_adaptive_compressor/gpt2/summary.csv line 2 -->
