# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `hybridkv`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor hybridkv \
    --output_dir results/unified_hybridkv
```
