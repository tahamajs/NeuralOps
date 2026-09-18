# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `progressive_token_pruning`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor progressive_token_pruning \
    --output_dir results/unified_progressive_token_pruning
```
