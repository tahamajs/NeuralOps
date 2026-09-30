# KD Run: kda_attention

**Date:** 2026-09-30 23:00:37
**Module:** `kda_attention`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 1
- Grad accumulation: 1 (effective batch = 1)
- Sequence length: 128
- Recovery steps: 1000
- Learning rate: 5.00e-06
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 2796.78
- Post-KD PPL: 156.64
- Recovery: 95.48%
- Post/Baseline ratio: 4.966x
- Step time: 458 ms
- Peak VRAM: 1501 MB
