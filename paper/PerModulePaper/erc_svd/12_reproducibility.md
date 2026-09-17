# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `erc_svd`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor erc_svd \
    --output_dir results/unified_erc_svd
```
