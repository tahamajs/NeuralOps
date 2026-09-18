# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `lola_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor lola_attention \
    --output_dir results/unified_lola_attention
```
