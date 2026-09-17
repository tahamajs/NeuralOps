# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `delta_kv`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor delta_kv \
    --output_dir results/unified_delta_kv
```
