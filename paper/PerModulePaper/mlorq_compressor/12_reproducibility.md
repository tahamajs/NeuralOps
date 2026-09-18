# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `mlorq_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor mlorq_compressor \
    --output_dir results/unified_mlorq_compressor
```
