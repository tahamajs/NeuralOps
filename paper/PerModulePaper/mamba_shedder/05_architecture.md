# 5. Architecture and Implementation Details

## 5.1 Submodule Hierarchy and Signal Flow
The `MambaShedderCompressor` architecture encapsulates both frozen base operations and optional trainable fine-tuning branches. Below is the structural hierarchy:

```
┌────────────────────────────────────────────────────────────────────────┐
│                                    MambaShedderCompressor              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Input Tensor X: [Batch, SeqLen, InFeatures]                          │
│         │                                                              │
│         ├────────────────────────────────────────┐                     │
│         │ (Primary Base Pathway)                 │ (LoRA Residual)     │
│         ▼                                        ▼                     │
│   ┌──────────────────────────┐             ┌───────────────────────┐   │
│   │ Down-Projection: V_r     │             │ LoRA Adapter A        │   │
│   │ Shape: (r, InFeatures)   │             │ Shape: (r_lora, In)   │   │
│   │ Status: Frozen           │             │ Trainable: True       │   │
│   └─────────────┬────────────┘             └───────────┬───────────┘   │
│                 │                                      │               │
│                 ▼                                      ▼               │
│   ┌──────────────────────────┐             ┌───────────────────────┐   │
│   │ Up-Projection: U_r       │             │ LoRA Adapter B        │   │
│   │ Shape: (OutFeatures, r)  │             │ Shape: (Out, r_lora)  │   │
│   │ Status: Frozen           │             │ Trainable: True       │   │
│   └─────────────┬────────────┘             └───────────┬───────────┘   │
│                 │                                      │               │
│                 │                                      │ * (alpha / r) │
│                 ▼                                      ▼               │
│              [ Base Output ]  <────── (+) ─────  [ Adapter Output ]    │
│                                                                        │
│   Output Tensor Y: [Batch, SeqLen, OutFeatures]                        │
└────────────────────────────────────────────────────────────────────────┘
```

## 5.2 Submodule Specification Matrix
The table below specifies the internal submodule configuration of `MambaShedderCompressor`:

| Submodule Name | Layer Type | Parameter Shape | Trainable Status | Memory Footprint (FP32) | Functional Role |
|---|---|---|---|---|---|
| `weight` / `V_r` | `nn.Parameter` / Buffer | `(r, d_in)` | Frozen (False) | $4 \cdot r \cdot d_{\text{in}}$ bytes | Low-rank down-projection or base weight |
| `U_r` | `nn.Parameter` / Buffer | `(d_out, r)` | Frozen (False) | $4 \cdot d_{\text{out}} \cdot r$ bytes | Low-rank up-projection expansion |
| `mask` | `torch.BoolTensor` Buffer | `(d_out, d_in)` | Non-trainable | $1 \cdot d_{\text{out}} \cdot d_{\text{in}}$ bytes | Hardware-aligned 2:4 or block sparsity mask |
| `lora_A` | `nn.Parameter` | `(r_lora, d_in)` | Trainable (True) | $4 \cdot r_{\text{lora}} \cdot d_{\text{in}}$ bytes | Fast adaptation gradient collector |
| `lora_B` | `nn.Parameter` | `(d_out, r_lora)` | Trainable (True) | $4 \cdot d_{\text{out}} \cdot r_{\text{lora}}$ bytes | Low-rank residual reconstruction projector |
| `bias` | `nn.Parameter` | `(d_out,)` | Optional (Frozen) | $4 \cdot d_{\text{out}}$ bytes | Projection affine bias term |

## 5.3 Cross-Backbone Adaptations
The module handles distinct idiosyncrasies across model backbones:
1. **DistilBERT (Flat Encoder Attention)**: Directly replaces `q_lin`, `k_lin`, `v_lin`, and `out_lin` within `MultiHeadSelfAttention`.
2. **RoBERTa (Nested Encoder Attention)**: Projections reside in `self.query`, `self.key`, `self.value`. The output projection `dense` is housed in the outer `BertSelfOutput` container. The engine detects nested structures and avoids duplicate output projection compression.
3. **GPT-2 (Causal Fused Conv1D)**: Attention weights are stored as a fused $d_{\text{in}} \times 3 d_{\text{in}}$ tensor with transposed 1D convolution layout. The engine transposes weights and extracts individual Q, K, V components.
4. **Qwen / LLaMA (GQA and RoPE)**: Key and value projections have reduced dimension ($H_{\text{kv}} \cdot d$). The engine aligns repeat factors and slices rotary position embeddings before computing compressed projections.
