# KD Run: compressed_attention_layer_energy

**Date:** 2026-10-01 13:56:19
**Module:** `compressed_attention_layer_energy`
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

- Pre-KD PPL: 601.81
- Post-KD PPL: 179.38
- Recovery: 74.08%
- Post/Baseline ratio: 5.687x
- Step time: 372 ms
- Peak VRAM: 1124 MB
