# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `flap`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor flap \
    --output_dir results/unified_flap
```
