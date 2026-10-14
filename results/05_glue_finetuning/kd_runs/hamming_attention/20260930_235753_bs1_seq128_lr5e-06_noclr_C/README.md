# KD Run: hamming_attention

**Date:** 2026-10-01 00:02:09
**Module:** `hamming_attention`
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

- Pre-KD PPL: 648.92
- Post-KD PPL: 83.59
- Recovery: 91.57%
- Post/Baseline ratio: 2.650x
- Step time: 460 ms
- Peak VRAM: 1496 MB
