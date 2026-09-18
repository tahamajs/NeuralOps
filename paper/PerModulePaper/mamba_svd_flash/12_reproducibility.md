# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `mamba_svd_flash`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor mamba_svd_flash \
    --output_dir results/unified_mamba_svd_flash
```
