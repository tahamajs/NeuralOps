# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `flat_llm`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor flat_llm \
    --output_dir results/unified_flat_llm
```
