# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `kda_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor kda_attention \
    --output_dir results/unified_kda_attention
```
