# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `replace`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor replace \
    --output_dir results/unified_replace
```
