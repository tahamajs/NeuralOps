# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `subspace_projection`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor subspace_projection \
    --output_dir results/unified_subspace_projection
```
