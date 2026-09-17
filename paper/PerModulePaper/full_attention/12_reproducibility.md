# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `full_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor full_attention \
    --output_dir results/unified_full_attention
```
