# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `mamba_inject`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor mamba_inject \
    --output_dir results/unified_mamba_inject
```
