# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `bonsai_pruner`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor bonsai_pruner \
    --output_dir results/unified_bonsai_pruner
```
