# KD Run: leka_v2

**Date:** 2026-09-30 17:40:06
**Module:** `leka_v2`
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

- Pre-KD PPL: 388.19
- Post-KD PPL: 140.60
- Recovery: 69.42%
- Post/Baseline ratio: 4.458x
- Step time: 976 ms
- Peak VRAM: 1267 MB
