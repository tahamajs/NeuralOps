# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `armor_pruner`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor armor_pruner \
    --output_dir results/unified_armor_pruner
```
