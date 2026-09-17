# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `kronecker_compressor`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor kronecker_compressor \
    --output_dir results/unified_kronecker_compressor
```
