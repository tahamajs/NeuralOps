# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `kronecker_tt_lora`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor kronecker_tt_lora \
    --output_dir results/unified_kronecker_tt_lora
```
