# Knowledge Distillation (KD) Benchmark Final Report

- **Task**: `glue:mrpc`
- **Teacher**: `textattack/bert-base-uncased-MRPC`
- **Total Modules**: 2
- **Timestamp**: 2026-10-13 12:06:04 UTC

## Module Results Summary

| Module | Status | Pre-KD Acc (%) | Best KD Acc (%) | Post-KD (Full) | Delta (%) | Duration (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `sink_attention` | **IMPROVED** | 48.00 | 81.00 | 77.50 | +29.50 | 15.2 |
| `naive_svd` | **IMPROVED** | 35.00 | 35.00 | 36.50 | +1.50 | 16.4 |
