# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `granite_hybrid`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor granite_hybrid \
    --output_dir results/unified_granite_hybrid
```
