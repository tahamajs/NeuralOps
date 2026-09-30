# KD Run: block_transformer_hiba

**Date:** 2026-09-30 22:28:56
**Module:** `block_transformer_hiba`
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

- Pre-KD PPL: 31.54
- Post-KD PPL: 31.76
- Recovery: 0.00%
- Post/Baseline ratio: 1.007x
- Step time: 340 ms
- Peak VRAM: 1453 MB
