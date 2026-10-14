# KD Run: aegis_v6_attn

**Date:** 2026-10-01 10:43:05
**Module:** `aegis_v6_attn`
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

- Pre-KD PPL: 212.20
- Post-KD PPL: 212.20
- Recovery: 0.00%
- Post/Baseline ratio: 6.728x
- Step time: 847 ms
- Peak VRAM: 1465 MB
