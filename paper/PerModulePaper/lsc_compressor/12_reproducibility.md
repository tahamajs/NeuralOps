# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `lsc_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor lsc_compressor \
    --output_dir results/unified_lsc_compressor
```
