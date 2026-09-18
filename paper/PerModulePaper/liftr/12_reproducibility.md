# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `liftr`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor liftr \
    --output_dir results/unified_liftr
```
