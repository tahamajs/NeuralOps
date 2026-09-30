# KD Run: primal_attention

**Date:** 2026-09-30 19:17:09
**Module:** `primal_attention`
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

- Pre-KD PPL: 31.70
- Post-KD PPL: 32.01
- Recovery: -193.75%
- Post/Baseline ratio: 1.015x
- Step time: 1299 ms
- Peak VRAM: 1482 MB
