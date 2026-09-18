# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `tomoe`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor tomoe \
    --output_dir results/unified_tomoe
```
