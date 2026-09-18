# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `vig_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor vig_attention \
    --output_dir results/unified_vig_attention
```
