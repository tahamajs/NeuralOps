# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `coopq_quantizer`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor coopq_quantizer \
    --output_dir results/unified_coopq_quantizer
```
