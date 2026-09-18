# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `zerokv_neo`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor zerokv_neo \
    --output_dir results/unified_zerokv_neo
```
