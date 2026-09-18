# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `tlinformer_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor tlinformer_attention \
    --output_dir results/unified_tlinformer_attention
```
