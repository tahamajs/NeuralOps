# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `lillama_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor lillama_compressor \
    --output_dir results/unified_lillama_compressor
```
