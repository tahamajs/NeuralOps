# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `hgrn2_outer`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor hgrn2_outer \
    --output_dir results/unified_hgrn2_outer
```
