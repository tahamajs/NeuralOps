# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `enlighten_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor enlighten_compressor \
    --output_dir results/unified_enlighten_compressor
```
