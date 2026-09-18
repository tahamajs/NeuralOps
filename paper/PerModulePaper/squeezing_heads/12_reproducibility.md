# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `squeezing_heads`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor squeezing_heads \
    --output_dir results/unified_squeezing_heads
```
