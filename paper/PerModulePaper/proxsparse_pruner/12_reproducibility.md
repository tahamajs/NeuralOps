# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `proxsparse_pruner`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor proxsparse_pruner \
    --output_dir results/unified_proxsparse_pruner
```
