# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `lsh_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor lsh_attention \
    --output_dir results/unified_lsh_attention
```
