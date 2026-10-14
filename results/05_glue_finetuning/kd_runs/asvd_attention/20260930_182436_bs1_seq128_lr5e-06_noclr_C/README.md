# KD Run: asvd_attention

**Date:** 2026-09-30 18:30:13
**Module:** `asvd_attention`
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

- Pre-KD PPL: 39.38
- Post-KD PPL: 34.82
- Recovery: 58.16%
- Post/Baseline ratio: 1.104x
- Step time: 488 ms
- Peak VRAM: 1162 MB
