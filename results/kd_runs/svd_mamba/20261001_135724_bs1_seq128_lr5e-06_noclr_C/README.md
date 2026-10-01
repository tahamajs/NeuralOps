# KD Run: svd_mamba

**Date:** 2026-10-01 14:07:51
**Module:** `svd_mamba`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 1
- Grad accumulation: 1 (effective batch = 1)
- Sequence length: 128
- Recovery steps: 500
- Learning rate: 5.00e-06
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 74.58
- Post-KD PPL: 47.11
- Recovery: 63.82%
- Post/Baseline ratio: 1.494x
- Step time: 951 ms
- Peak VRAM: 2353 MB
