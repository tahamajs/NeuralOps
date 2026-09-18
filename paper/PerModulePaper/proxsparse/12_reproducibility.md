# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `proxsparse`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor proxsparse \
    --output_dir results/unified_proxsparse
```
