# KD Run: res_svd

**Date:** 2026-09-30 19:04:27
**Module:** `res_svd`
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

- Pre-KD PPL: 277.40
- Post-KD PPL: 115.30
- Recovery: 65.93%
- Post/Baseline ratio: 3.656x
- Step time: 409 ms
- Peak VRAM: 1149 MB
