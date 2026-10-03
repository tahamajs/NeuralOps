# KD Run: gated_delta_net

**Date:** 2026-09-30 22:48:40
**Module:** `gated_delta_net`
**Methods:** C
**Train scope:** `mamba`

## Configuration

- Batch size: 4
- Grad accumulation: 8 (effective batch = 32)
- Sequence length: 128
- Recovery steps: 300
- Learning rate: 5.00e-06
- KL temperature: 2.0
- LR schedule: constant (no cosine)
- Cache max texts: 500

## Results

### Method C

- Pre-KD PPL: 784.04
- Post-KD PPL: 177.42
- Recovery: 80.61%
- Post/Baseline ratio: 5.625x
- Step time: 3169 ms
- Peak VRAM: 3940 MB
