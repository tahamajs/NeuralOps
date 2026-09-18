# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `svd_proj_attn`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor svd_proj_attn \
    --output_dir results/unified_svd_proj_attn
```
