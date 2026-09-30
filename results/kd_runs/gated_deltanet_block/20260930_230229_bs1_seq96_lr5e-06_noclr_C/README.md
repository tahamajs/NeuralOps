# KD Run: gated_deltanet_block

**Date:** 2026-09-30 23:05:35
**Module:** `gated_deltanet_block`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 1
- Grad accumulation: 1 (effective batch = 1)
- Sequence length: 96
- Recovery steps: 300
- Learning rate: 5.00e-06
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 1332.05
- Post-KD PPL: 454.39
- Recovery: 67.61%
- Post/Baseline ratio: 13.361x
- Step time: 531 ms
- Peak VRAM: 1748 MB
