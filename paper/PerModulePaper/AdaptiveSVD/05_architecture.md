# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support Matrix

Adaptive SVD supports the following attention variants:

| Architecture | Class | Nested? | RoPE? | GQA? | T5 bias? | Verified |
|---|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No | ✅ |
| GPT-2 | `GPT2Attention` | No | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No | partial |
| T5 | `T5Attention` | No | No | No | **Yes** | not tested |

## 5.2 Weight Extraction

### 5.2.1 DistilBERT (Flat)

Direct extraction:

```python
W_Q = src.q_lin.weight  # Linear(768, 768)
W_K = src.k_lin.weight
W_V = src.v_lin.weight
W_O = src.out_lin.weight
```

All four projections are inside `MultiHeadSelfAttention`. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

Extract from inner `BertSelfAttention`:

```python
W_Q = src.self.query.weight
W_K = src.self.key.weight
W_V = src.self.value.weight
# W_O is not extracted — outer BertSelfOutput.dense applies it
```

`is_nested_encoder = True`. During forward, the compressed `context` is 
returned without applying `out_proj`; the outer `BertSelfOutput.dense` 
applies its own (unchanged) projection.

**Consequence.** In RoBERTa, the output projection $W_O$ is **not compressed**, 
only $W_Q, W_K, W_V$.

### 5.2.3 GPT-2 (Fused Conv1D)

GPT-2 fuses Q, K, V into a single `c_attn` Conv1D:

```python
fused = src.c_attn.weight.T  # (3*768, 768) after transpose
W_Q, W_K, W_V = fused.chunk(3, dim=0)
W_O = src.c_proj.weight.T
```

The transpose handles the HuggingFace Conv1D layout: it stores weights as 
`(in_features, out_features)` instead of the standard `(out_features, in_features)`.

### 5.2.4 T5 (Relative Attention Bias)

T5 uses `relative_attention_bias` (an `nn.Embedding`) instead of learned 
position embeddings:

```python
self.relative_attention_bias = src.relative_attention_bias
self.has_relative_attention_bias = True
```

The bias is preserved as a separate module and added to attention scores 
in forward. The bucket function handles bidirectional (encoder) and causal 
(decoder) modes.

### 5.2.5 Qwen3.5 (GQA + RoPE)

Qwen3.5 uses GQA with $H_{kv} < H$:

```python
W_Q = src.q_proj.weight  # (H * d_h, d) = (2048, 1024)
W_K = src.k_proj.weight  # (H_kv * d_h, d) = (512, 1024)
W_V = src.v_proj.weight  # (512, 1024)
W_O = src.o_proj.weight  # (1024, 2048)
```

RoPE is applied after projection:

```python
Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
```

## 5.3 The `AdaptiveSVDLinear` Module

```python
class AdaptiveSVDLinear(nn.Module):
    def __init__(self, in_features, out_features, svd_rank, residual_rank=16, ...):
        super().__init__()
        self.svd_rank = min(svd_rank, in_features, out_features)
        self.residual_rank = min(residual_rank, in_features, out_features)
        
        # Frozen SVD base
        self.U = nn.Parameter(
            torch.empty(out_features, self.svd_rank),
            requires_grad=False
        )
        self.V = nn.Parameter(
            torch.empty(self.svd_rank, in_features),
            requires_grad=False
        )
        
        # Trainable residual
        if self.residual_rank > 0:
            self.lora_B = nn.Parameter(
                torch.empty(out_features, self.residual_rank)
            )
            self.lora_A = nn.Parameter(
                torch.empty(self.residual_rank, in_features)
            )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        
        self.reset_parameters()
    
    def forward(self, x):
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)
        if self.lora_A is not None:
            Ax = F.linear(x, self.lora_A)
            out = out + F.linear(Ax, self.lora_B)
        if self.bias is not None:
            out = out + self.bias
        return out
```

**Parameter count:** $r_P(d_\text{in} + d_\text{out}) + r_L(d_\text{in} + d_\text{out}) + d_\text{out}$.

## 5.4 Injection Algorithm

**Algorithm 2:** Inject Adaptive SVD into a full model
```
Input:  model M, config c, hyperparameters τ, r_L
Output: M with attention replaced

1:  for ℓ in transformer_layers(M) do
2:      src ← find_attention(ℓ)                   ▷ Old attention
3:      if src is None: continue
4:      new ← AdaptiveSVDAttention(c, τ, r_L, pretrained_attn=src)
5:      new.copy_weights(src)                      ▷ SVD + adapter
6:      replace_attention(ℓ, new)
7: end for
8: return M
```

**Layer paths:**

| Model | Layer path | Attention path |
|---|---|---|
| DistilBERT | `distilbert.transformer.layer[i]` | `.attention` |
| RoBERTa | `roberta.encoder.layer[i]` | `.attention.self` |
| GPT-2 | `transformer.h[i]` | `.attn` |
| T5 | `encoder.block[i].layer[0]` | `.SelfAttention` |

## 5.5 Handling Special Cases

### 5.5.1 T5 Relative Attention Bias

Forward:

```python
position_bias = kwargs.get("position_bias", None)
if position_bias is None and self.has_relative_attention_bias:
    position_bias = self.compute_bias(N_q, N_k, device=hidden_states.device)
```

The `compute_bias` function uses `_relative_position_bucket` to map relative 
positions to buckets, then looks up the bias embedding.

### 5.5.2 GQA Expansion

```python
if self.kv_repeat > 1:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

### 5.5.3 RoPE

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
```

### 5.5.4 Causal Masking

```python
if is_causal:
    causal_tril = torch.tril(torch.ones((N_q, N_k), device=...))
    scores = scores.masked_fill(~causal_tril.unsqueeze(0).unsqueeze(0), -10000.0)
```

## 5.6 Registry

```python
@ATTENTION_REGISTRY.register("adaptive_svd_attn")
@ATTENTION_REGISTRY.register("adaptive_svd")
@ATTENTION_REGISTRY.register("adaptive_svd_attention")
class AdaptiveSVDAttention(BaseAttention):
    ...
```

## 5.7 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Return None |
| Shape mismatch | Assertion | Skip layer |
| Near-zero Frobenius | Check < 1e-4 | Skip layer |
| NaN in SVD | `torch.isnan` | Fallback |
| T5 bias mismatch | Embedding shape | Skip |

## 5.8 Numerical Precision

| Component | Precision |
|---|---|
| SVD | FP32 |
| Factor storage | Model dtype |
| Forward | Model dtype |
| Adapter init | FP32 then cast |

## 5.9 Compatibility

### 5.9.1 LoRA Wrappers

If the source module is a `LoRALinear` wrapper, Adaptive SVD extracts the 
inner `base_layer` for SVD and preserves the outer LoRA parameters:

```python
target_mod = getattr(module, "base_layer", module)
```

### 5.9.2 Encoder-Decoder

For T5, the same `AdaptiveSVDAttention` is used in both encoder and decoder 
blocks. The decoder side has `is_decoder = True`, which enables causal masking.
