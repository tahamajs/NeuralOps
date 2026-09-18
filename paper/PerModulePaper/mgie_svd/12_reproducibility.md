# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `mgie_svd`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor mgie_svd \
    --output_dir results/unified_mgie_svd
```
