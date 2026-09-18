# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `samba_moe`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor samba_moe \
    --output_dir results/unified_samba_moe
```
