# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `cr_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor cr_compressor \
    --output_dir results/unified_cr_compressor
```
