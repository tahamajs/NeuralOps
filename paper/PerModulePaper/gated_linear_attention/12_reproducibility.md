# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `gated_linear_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor gated_linear_attention \
    --output_dir results/unified_gated_linear_attention
```
