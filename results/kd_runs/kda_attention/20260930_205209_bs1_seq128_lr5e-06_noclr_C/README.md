# KD Run: kda_attention

**Date:** 2026-09-30 20:56:44
**Module:** `kda_attention`
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

- Pre-KD PPL: 2796.78
- Post-KD PPL: 189.05
- Recovery: 94.30%
- Post/Baseline ratio: 5.994x
- Step time: 492 ms
- Peak VRAM: 1501 MB
