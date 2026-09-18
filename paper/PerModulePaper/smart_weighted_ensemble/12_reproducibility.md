# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `smart_weighted_ensemble`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor smart_weighted_ensemble \
    --output_dir results/unified_smart_weighted_ensemble
```
