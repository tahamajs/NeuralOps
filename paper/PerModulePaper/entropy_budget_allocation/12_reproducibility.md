# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `entropy_budget_allocation`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor entropy_budget_allocation \
    --output_dir results/unified_entropy_budget_allocation
```
