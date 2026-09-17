# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `kmeans_quantize`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor kmeans_quantize \
    --output_dir results/unified_kmeans_quantize
```
