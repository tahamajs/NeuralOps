# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `slim_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor slim_compressor \
    --output_dir results/unified_slim_compressor
```
