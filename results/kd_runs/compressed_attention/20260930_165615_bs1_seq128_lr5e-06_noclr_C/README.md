# KD Run: compressed_attention

**Date:** 2026-09-30 17:03:19
**Module:** `compressed_attention`
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

- Pre-KD PPL: 165.59
- Post-KD PPL: 43.20
- Recovery: 91.30%
- Post/Baseline ratio: 1.370x
- Step time: 389 ms
- Peak VRAM: 1255 MB
