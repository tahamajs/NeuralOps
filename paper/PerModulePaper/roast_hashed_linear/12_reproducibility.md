# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `roast_hashed_linear`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor roast_hashed_linear \
    --output_dir results/unified_roast_hashed_linear
```
