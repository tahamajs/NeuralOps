# KD Run: kda_attention

**Date:** 2026-09-30 23:31:25
**Module:** `kda_attention`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 2
- Grad accumulation: 4 (effective batch = 8)
- Sequence length: 128
- Recovery steps: 1000
- Learning rate: 5.00e-05
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 2796.78
- Post-KD PPL: 54.56
- Recovery: 99.17%
- Post/Baseline ratio: 1.730x
- Step time: 1371 ms
- Peak VRAM: 1942 MB
