# KD Run: aegis_v5

**Date:** 2026-09-30 22:17:19
**Module:** `aegis_v5`
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

- Pre-KD PPL: 211.86
- Post-KD PPL: 98.77
- Recovery: 62.72%
- Post/Baseline ratio: 3.132x
- Step time: 781 ms
- Peak VRAM: 1226 MB
