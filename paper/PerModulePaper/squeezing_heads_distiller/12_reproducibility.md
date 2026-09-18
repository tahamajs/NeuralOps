# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `squeezing_heads_distiller`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor squeezing_heads_distiller \
    --output_dir results/unified_squeezing_heads_distiller
```
