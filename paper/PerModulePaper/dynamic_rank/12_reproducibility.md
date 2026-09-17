# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `dynamic_rank`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor dynamic_rank \
    --output_dir results/unified_dynamic_rank
```
