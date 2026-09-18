# 12. Reproducibility Guide

## 12.1 Compression Execution Command
```bash
python -u Codes/framework/run_compressor.py - model distilbert-base-uncased - compressor aa_svd_compressor - rank 288 - ema_beta 0.9 - max_error 0.50
```

## 12.2 Verification Test Command
```bash
pytest Codes/framework/tests/test_adaptive_svd_verification.py -v
```
