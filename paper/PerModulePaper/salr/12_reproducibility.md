# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `salr`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor salr \
    --output_dir results/unified_salr
```
