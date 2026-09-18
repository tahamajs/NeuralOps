# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `squeezing_heads_distill`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor squeezing_heads_distill \
    --output_dir results/unified_squeezing_heads_distill
```
