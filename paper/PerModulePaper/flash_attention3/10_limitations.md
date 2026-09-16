# 10. Limitations

## 10.1 Hardware Dependency

The FlashAttention-3 kernel requires **Hopper** architecture (H100, H200). 
On Ampere, Turing, or older GPUs, the kernel is unavailable and our code 
falls back to SDPA. **No speedup is realized.**

## 10.2 No Parameter Reduction

FlashAttention-3 is an **algorithm**, not a compression method. Parameter 
count increases by ~{{D_PARAM_PCT}}% due to LoRA adapters. If your goal is 
to reduce model size, this module is not the right choice.

## 10.3 FP8 Precision

FP8 has limited mantissa precision. We have only tested FP8 on SST-2; 
behavior on other tasks is untested.

## 10.4 Library Dependency

We require `flash_attn` or `flash_attn_interface` for FA3. If neither is 
installed, only SDPA is available.

## 10.5 Task Coverage

Only SST-2 is evaluated. Long-context generation and seq2seq tasks are 
untested.

## 10.6 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |
| Hopper (H100) | Not directly tested here |

## 10.7 Hardware

Our benchmark GPU is RTX 4070 (Ampere). We do not test on Hopper here.

## 10.8 Cross-Attention Not Fully Tested

We implement cross-attention but do not test T5 or BART.

## 10.9 Single-Seed Results

Multi-seed would strengthen conclusions.

## 10.10 No Novel Algorithm

We do not propose a new attention algorithm — we integrate FlashAttention-3.

## 10.11 Code Quality

Research-grade:
- Fallback exceptions are broadly caught
- No unit tests for edge cases
- Some detection orders differ from other modules

## 10.12 Ethical Considerations

Attention optimization is dual-use. See Appendix E.

---