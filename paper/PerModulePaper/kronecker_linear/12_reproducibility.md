# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `kronecker_linear`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor kronecker_linear \
    --output_dir results/unified_kronecker_linear
```
