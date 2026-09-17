# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `factorised_matrix_state`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor factorised_matrix_state \
    --output_dir results/unified_factorised_matrix_state
```
