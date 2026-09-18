# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `saes_svd_v2`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor saes_svd_v2 \
    --output_dir results/unified_saes_svd_v2
```
