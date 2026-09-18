# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `uniql_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor uniql_compressor \
    --output_dir results/unified_uniql_compressor
```
