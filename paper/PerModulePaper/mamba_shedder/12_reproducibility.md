# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `mamba_shedder`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor mamba_shedder \
    --output_dir results/unified_mamba_shedder
```
