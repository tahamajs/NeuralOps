# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `hybrid_svd_layer`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor hybrid_svd_layer \
    --output_dir results/unified_hybrid_svd_layer
```
