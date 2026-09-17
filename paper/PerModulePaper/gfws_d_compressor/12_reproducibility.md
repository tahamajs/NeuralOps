# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `gfws_d_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor gfws_d_compressor \
    --output_dir results/unified_gfws_d_compressor
```
