# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `samba_m3`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor samba_m3 \
    --output_dir results/unified_samba_m3
```
