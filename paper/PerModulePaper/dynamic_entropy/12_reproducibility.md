# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `dynamic_entropy`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor dynamic_entropy \
    --output_dir results/unified_dynamic_entropy
```
