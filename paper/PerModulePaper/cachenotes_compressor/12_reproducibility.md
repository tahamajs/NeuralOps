# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `cachenotes_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor cachenotes_compressor \
    --output_dir results/unified_cachenotes_compressor
```
