# KD Run: aegis_v6

**Date:** 2026-09-30 18:12:33
**Module:** `aegis_v6`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 1
- Grad accumulation: 1 (effective batch = 1)
- Sequence length: 128
- Recovery steps: 10
- Learning rate: 5.00e-06
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 157.78
- Post-KD PPL: 157.78
- Recovery: 0.00%
- Post/Baseline ratio: 5.003x
- Step time: 3551 ms
- Peak VRAM: 1470 MB
