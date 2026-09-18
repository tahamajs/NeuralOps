# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `soka`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor soka \
    --output_dir results/unified_soka
```
