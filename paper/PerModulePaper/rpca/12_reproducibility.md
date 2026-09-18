# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `rpca`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor rpca \
    --output_dir results/unified_rpca
```
