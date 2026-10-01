# KD Run: hybrid_attention

**Date:** 2026-10-01 00:16:16
**Module:** `hybrid_attention`
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
- Post-KD PPL: 1198.81
- Recovery: 0.00%
- Post/Baseline ratio: 38.009x
- Step time: 506 ms
- Peak VRAM: 1639 MB
