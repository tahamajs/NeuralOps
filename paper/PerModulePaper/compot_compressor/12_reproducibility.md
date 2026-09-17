# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `compot_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor compot_compressor \
    --output_dir results/unified_compot_compressor
```
