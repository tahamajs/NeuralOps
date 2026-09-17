# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `frandi`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor frandi \
    --output_dir results/unified_frandi
```
