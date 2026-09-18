# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `replaceme_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor replaceme_compressor \
    --output_dir results/unified_replaceme_compressor
```
