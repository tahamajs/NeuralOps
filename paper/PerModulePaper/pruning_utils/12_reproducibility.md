# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `pruning_utils`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor pruning_utils \
    --output_dir results/unified_pruning_utils
```
