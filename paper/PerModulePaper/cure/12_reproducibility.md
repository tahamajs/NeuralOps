# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `cure`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor cure \
    --output_dir results/unified_cure
```
