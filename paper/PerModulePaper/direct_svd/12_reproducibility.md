# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `direct_svd`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor direct_svd \
    --output_dir results/unified_direct_svd
```
