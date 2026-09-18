# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `sparsegpt`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor sparsegpt \
    --output_dir results/unified_sparsegpt
```
