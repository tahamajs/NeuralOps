# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `slicegpt`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor slicegpt \
    --output_dir results/unified_slicegpt
```
