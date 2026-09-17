# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `ast_pruner`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor ast_pruner \
    --output_dir results/unified_ast_pruner
```
