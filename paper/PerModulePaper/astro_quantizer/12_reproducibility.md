# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `astro_quantizer`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor astro_quantizer \
    --output_dir results/unified_astro_quantizer
```
