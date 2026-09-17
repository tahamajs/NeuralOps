# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `dobi`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor dobi \
    --output_dir results/unified_dobi
```
