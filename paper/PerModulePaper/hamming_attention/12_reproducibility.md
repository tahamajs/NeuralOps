# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `hamming_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor hamming_attention \
    --output_dir results/unified_hamming_attention
```
