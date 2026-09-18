# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `optimal_subspace_projection`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor optimal_subspace_projection \
    --output_dir results/unified_optimal_subspace_projection
```
