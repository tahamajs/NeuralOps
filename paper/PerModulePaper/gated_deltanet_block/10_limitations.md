<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Fresh LayerNorm

The block's LayerNorm is initialized from $\gamma = 1, \beta = 0$, which 
does not match the pretrained model's distribution. This is the **primary** 
limitation.

## 10.2 Silent Weight Transfer Failure

The `try/except` in `copy_weights` swallows five exception types. Any 
failure leaves the attention with **random weights** and no warning.

## 10.3 Non-Causal MHA for GPT-2

The `use_full_attention=True` path uses `nn.MultiheadAttention` without 
causal masking. **This is incorrect for GPT-2.**

## 10.4 Nested Encoder Inconsistency

For RoBERTa, the block returns raw output without LayerNorm. This is 
different from the behavior on DistilBERT.

## 10.5 Slow GatedDeltaNet Path

The `use_full_attention=False` path uses GatedDeltaNet, which is 
**~84.5× slower** than baseline.

## 10.6 OOM on 12-Layer Models

RoBERTa and GPT-2 fail with CUDA OOM at batch_size=8.

## 10.7 No Multi-Seed

Single seed.

## 10.8 Task Coverage

Only SST-2. No generation tasks tested (important for causal masking issue).

## 10.9 Model Coverage

Only DistilBERT completes. Larger models fail.

## 10.10 Parameter Overhead

LayerNorm adds $2d$ params per block (small). Attention + LoRA add more.

## 10.11 Code Quality

- Silent `try/except` masks bugs
- No logging of weight transfer status
- Non-causal MHA for GPT-2
- Nested encoder path differs from flat path
- The `if attn_out.shape[-1] != x.shape[-1]` padding logic may silently 
  pass through wrong dimensions

## 10.12 What This Paper Is Not

- Not a compression method
- Not a speedup method
- Not a recommended replacement for baseline transformer blocks

---