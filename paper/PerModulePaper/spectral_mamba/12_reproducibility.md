# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `spectral_mamba`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor spectral_mamba \
    --output_dir results/unified_spectral_mamba
```
