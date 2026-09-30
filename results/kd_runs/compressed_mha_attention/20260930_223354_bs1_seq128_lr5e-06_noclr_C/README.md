# KD Run: compressed_mha_attention

**Date:** 2026-09-30 22:37:08
**Module:** `compressed_mha_attention`
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

- Pre-KD PPL: 601.87
- Post-KD PPL: 176.91
- Recovery: 74.51%
- Post/Baseline ratio: 5.609x
- Step time: 335 ms
- Peak VRAM: 1111 MB
