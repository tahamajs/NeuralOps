# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `hashed_net_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor hashed_net_compressor \
    --output_dir results/unified_hashed_net_compressor
```
