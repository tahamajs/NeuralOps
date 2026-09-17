# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `compressed_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor compressed_attention \
    --output_dir results/unified_compressed_attention
```
