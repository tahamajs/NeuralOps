# KD Run: hla_attention

**Date:** 2026-10-01 00:10:25
**Module:** `hla_attention`
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

- Pre-KD PPL: 3229.08
- Post-KD PPL: 1750.78
- Recovery: 46.23%
- Post/Baseline ratio: 55.510x
- Step time: 555 ms
- Peak VRAM: 1496 MB
