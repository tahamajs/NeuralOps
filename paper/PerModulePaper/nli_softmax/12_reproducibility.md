# 12. Reproducibility Guide

To reproduce the benchmark evaluations for `nli_softmax`:

```bash
python -u Codes/framework/run_compressor.py \
    --model distilbert-base-uncased \
    --compressor nli_softmax \
    --output_dir results/unified_nli_softmax
```
