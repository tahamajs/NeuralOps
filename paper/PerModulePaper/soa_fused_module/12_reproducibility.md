# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `soa_fused_module`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor soa_fused_module \
    --output_dir results/unified_soa_fused_module
```
