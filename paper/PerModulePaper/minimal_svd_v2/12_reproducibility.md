# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `minimal_svd_v2`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor minimal_svd_v2 \
    --output_dir results/unified_minimal_svd_v2
```
