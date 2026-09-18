# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `optimized_hybrid_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor optimized_hybrid_attention \
    --output_dir results/unified_optimized_hybrid_attention
```
