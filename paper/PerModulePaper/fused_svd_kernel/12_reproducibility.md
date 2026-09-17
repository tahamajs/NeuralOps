# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `fused_svd_kernel`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor fused_svd_kernel \
    --output_dir results/unified_fused_svd_kernel
```
