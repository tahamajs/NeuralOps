# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `weighted_ensemble`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor weighted_ensemble \
    --output_dir results/unified_weighted_ensemble
```
