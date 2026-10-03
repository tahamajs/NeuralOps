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

AdaSplash supports the following attention variants:

| Architecture | Class | Nested? | RoPE? | GQA? | T5 bias? | Verified |
|---|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No | ✅ |
| GPT-2 | `GPT2Attention` | No | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No | partial |
| T5 | `T5Attention` | No | No | No | Yes | not tested |

## 5.2 Weight Extraction

### 5.2.1 DistilBERT (Flat)

Direct extraction:

```python
W_Q = src.q_lin.weight  # Linear(768, 768)
W_K = src.k_lin.weight
W_V = src.v_lin.weight
W_O = src.out_lin.weight
```

`is_nested_encoder = False`. All four projections compressed.

### 5.2.2 RoBERTa (Nested)

Extract from `self`:

```python
W_Q = src.self.query.weight
W_K = src.self.key.weight
W_V = src.self.value.weight
# W_O is not extracted — outer output.dense applies it
```

`is_nested_encoder = True`. During forward, `context` is returned without 
`out_proj`; the outer `BertSelfOutput.dense` applies its projection.

### 5.2.3 GPT-2 (Fused Conv1D)

Split the fused `c_attn` weight:

```python
fused = src.c_attn.weight.T  # (2304, 768)
W_Q, W_K, W_V = fused.chunk(3, dim=0)
W_O = src.c_proj.weight.T
```

Handles the HuggingFace Conv1D layout `(in, out)`.

### 5.2.4 Qwen3.5 (GQA + RoPE)

Direct extraction with GQA awareness:

```python
W_Q = src.q_proj.weight  # (H * d_h, d)
W_K = src.k_proj.weight  # (H_kv * d_h, d)
W_V = src.v_proj.weight
W_O = src.o_proj.weight
```

RoPE applied after projection via `position_embeddings`.

## 5.3 The `AdaSplashLinear` Module

```python
class AdaSplashLinear(nn.Module):
    def __init__(self, in_features, out_features, lora_rank=16, bias=True):
        super().__init__()
        self.lora_rank = min(lora_rank, in_features, out_features)

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)

        # Trainable LoRA
        if self.lora_rank > 0:
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank))
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features))

        self.reset_parameters()

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_A is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out
```

## 5.4 Injection Algorithm

**Algorithm 3:** Inject AdaSplash into a model
```
Input:  model M, config c, hyperparameters α, W, r_L
Output: M with attention replaced

1:  for ℓ in transformer_layers(M) do
2:      src ← find_attention(ℓ)
3:      if src is None: continue
4:      new ← AdaSplashAttention(c, α, W, r_L, pretrained_attn=src)
5:      new.copy_weights(src)
6:      replace_attention(ℓ, new)
7: end for
8: return M
```

## 5.5 Handling Special Cases

### 5.5.1 Attention Sinks

For causal decoders, we preserve the first $S = 4$ tokens regardless of 
splash mask:

```python
splash_local = ((dist >= 0) & (dist <= W)) | (k_idx < 4)
```

### 5.5.2 GQA Expansion

```python
if self.kv_repeat > 1:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

### 5.5.3 RoPE

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    q, k = apply_rotary_emb(q, k, cos, sin)
```

### 5.5.4 Causal Masking

```python
if is_decoder:
    causal_tril = torch.tril(torch.ones((L_q, L_k), device=device))
    logits = logits.masked_fill(~causal_tril, -10000)
```

### 5.5.5 Combined Masking

The causal mask and splash mask are combined:

```python
if is_decoder and W > 0:
    dist = q_idx - k_idx
    splash = ((dist >= 0) & (dist <= W)) | (k_idx < 4)
    causal_and_splash = splash  # already includes causality via dist >= 0
    logits = logits.masked_fill(~causal_and_splash, -10000)
```

## 5.6 Registry

```python
@ATTENTION_REGISTRY.register("adasplash")
@ATTENTION_REGISTRY.register("adasplash_attention")
class AdaSplashAttention(BaseAttention): ...
```

## 5.7 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Return None |
| Shape mismatch | Assertion in `copy_weights` | Skip layer |
| Near-zero Frobenius | Check < 1e-4 | Raise |
| Entmax support empty | `sum(p) < 1e-7` | Softmax fallback |
| NaN in scores | `torch.isnan` | Mask to -10000 |

## 5.8 Numerical Precision

| Component | Precision |
|---|---|
| Scores | Model dtype |
| Entmax bisection | FP32 internally |
| Attention output | Model dtype |
| Base weights | Model dtype |
| LoRA | Model dtype |

## 5.9 Compatibility

### 5.9.1 Encoder-Decoder Models

For T5, the same class handles encoder (bidirectional, no splash) and decoder 
(causal, splash). The `is_decoder` flag switches behavior.

### 5.9.2 LoRA Wrappers

If the source module is a `LoRALinear`, AdaSplash unwraps to `base_layer` and 
preserves the outer LoRA.
