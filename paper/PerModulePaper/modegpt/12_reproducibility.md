# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `modegpt`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor modegpt \
    --output_dir results/unified_modegpt
```
