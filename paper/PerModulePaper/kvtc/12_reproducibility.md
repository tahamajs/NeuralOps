# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `kvtc`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor kvtc \
    --output_dir results/unified_kvtc
```
