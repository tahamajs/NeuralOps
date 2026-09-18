# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `slicegpt_pca`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor slicegpt_pca \
    --output_dir results/unified_slicegpt_pca
```
