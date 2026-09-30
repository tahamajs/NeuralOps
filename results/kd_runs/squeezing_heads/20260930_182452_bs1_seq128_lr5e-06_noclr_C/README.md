# KD Run: squeezing_heads

**Date:** 2026-09-30 18:29:09
**Module:** `squeezing_heads`
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
- Post-KD PPL: 31.83
- Recovery: 0.00%
- Post/Baseline ratio: 1.009x
- Step time: 451 ms
- Peak VRAM: 1454 MB
