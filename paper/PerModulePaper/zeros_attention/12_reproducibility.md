# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `zeros_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor zeros_attention \
    --output_dir results/unified_zeros_attention
```
