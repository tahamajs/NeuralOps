# KD Run: decay_linear_attention

**Date:** 2026-09-30 22:46:32
**Module:** `decay_linear_attention`
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

- Pre-KD PPL: 2492.37
- Post-KD PPL: 131.84
- Recovery: 95.92%
- Post/Baseline ratio: 4.180x
- Step time: 968 ms
- Peak VRAM: 1492 MB
