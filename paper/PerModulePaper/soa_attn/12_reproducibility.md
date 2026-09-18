# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `soa_attn`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor soa_attn \
    --output_dir results/unified_soa_attn
```
