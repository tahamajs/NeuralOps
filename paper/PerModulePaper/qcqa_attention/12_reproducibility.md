# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `qcqa_attention`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor qcqa_attention \
    --output_dir results/unified_qcqa_attention
```
