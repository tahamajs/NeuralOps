# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `faq_quantizer`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor faq_quantizer \
    --output_dir results/unified_faq_quantizer
```
