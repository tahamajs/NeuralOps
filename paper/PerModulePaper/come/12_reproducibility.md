# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `come`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor come \
    --output_dir results/unified_come
```
