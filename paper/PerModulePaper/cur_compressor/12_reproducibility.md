# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `cur_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor cur_compressor \
    --output_dir results/unified_cur_compressor
```
