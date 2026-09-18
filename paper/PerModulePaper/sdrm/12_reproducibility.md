# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `sdrm`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor sdrm \
    --output_dir results/unified_sdrm
```
