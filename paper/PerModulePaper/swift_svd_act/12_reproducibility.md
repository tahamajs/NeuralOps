# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `swift_svd_act`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor swift_svd_act \
    --output_dir results/unified_swift_svd_act
```
