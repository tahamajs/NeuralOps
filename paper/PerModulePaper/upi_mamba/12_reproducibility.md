# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `upi_mamba`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor upi_mamba \
    --output_dir results/unified_upi_mamba
```
