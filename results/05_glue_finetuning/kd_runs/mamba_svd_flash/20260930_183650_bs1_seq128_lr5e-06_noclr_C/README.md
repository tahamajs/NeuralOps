# KD Run: mamba_svd_flash

**Date:** 2026-09-30 18:39:41
**Module:** `mamba_svd_flash`
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

- Pre-KD PPL: 31.54
- Post-KD PPL: 32.03
- Recovery: 0.00%
- Post/Baseline ratio: 1.016x
- Step time: 273 ms
- Peak VRAM: 1419 MB
