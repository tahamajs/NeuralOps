# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `jamba`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor jamba \
    --output_dir results/unified_jamba
```
