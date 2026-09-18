# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `rotational_slicegpt_int4`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor rotational_slicegpt_int4 \
    --output_dir results/unified_rotational_slicegpt_int4
```
