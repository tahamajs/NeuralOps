# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `galerkin_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor galerkin_attention \
    --output_dir results/unified_galerkin_attention
```
