# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `hla_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor hla_attention \
    --output_dir results/unified_hla_attention
```
