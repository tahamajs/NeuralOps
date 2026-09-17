# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `fusion_attn`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor fusion_attn \
    --output_dir results/unified_fusion_attn
```
