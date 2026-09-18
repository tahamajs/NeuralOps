# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `wgqa`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor wgqa \
    --output_dir results/unified_wgqa
```
