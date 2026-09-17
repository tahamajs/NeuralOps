# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `halo_regla`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor halo_regla \
    --output_dir results/unified_halo_regla
```
