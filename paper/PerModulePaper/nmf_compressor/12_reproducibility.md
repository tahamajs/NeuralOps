# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `nmf_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor nmf_compressor \
    --output_dir results/unified_nmf_compressor
```
