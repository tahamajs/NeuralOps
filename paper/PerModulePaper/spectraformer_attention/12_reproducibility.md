# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `spectraformer_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor spectraformer_attention \
    --output_dir results/unified_spectraformer_attention
```
