# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `svd_linear_decay`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor svd_linear_decay \
    --output_dir results/unified_svd_linear_decay
```
