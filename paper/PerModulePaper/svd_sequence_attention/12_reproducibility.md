# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `svd_sequence_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor svd_sequence_attention \
    --output_dir results/unified_svd_sequence_attention
```
