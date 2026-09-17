# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `grecodec`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor grecodec \
    --output_dir results/unified_grecodec
```
