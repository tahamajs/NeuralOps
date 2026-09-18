<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Architecture integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Extraction

### 5.2.1 DistilBERT

```python
extracted["q"] = src.q_lin.weight
extracted["k"] = src.k_lin.weight
extracted["v"] = src.v_lin.weight
extracted["o"] = src.out_lin.weight
```

### 5.2.2 RoBERTa (nested)

```python
s = src.self
extracted["q"] = s.query.weight
extracted["k"] = s.key.weight
extracted["v"] = s.value.weight
extracted["o"] = src.output.dense.weight
```

### 5.2.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.t()
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = q_w
extracted["k"] = k_w
extracted["v"] = v_w
extracted["o"] = src.c_proj.weight.t()
```

### 5.2.4 Qwen (GQA + RoPE)

```python
extracted["q"] = src.q_proj.weight
extracted["k"] = src.k_proj.weight
extracted["v"] = src.v_proj.weight
extracted["o"] = src.o_proj.weight
```

## 5.3 GQA Adaptation

For Qwen-style models where $H_{kv} < H$, the extracted K/V weights are 
adapted:

```python
def _adapt_to_kv_size(weight, bias):
    if weight.shape[0] == self.kv_head_size:
        return weight, bias
    if weight.shape[0] > self.kv_head_size and weight.shape[0] % self.kv_head_size == 0:
        rep = weight.shape[0] // self.kv_head_size
        weight = weight.reshape(self.num_kv_heads, rep, -1).mean(dim=1)
        if bias is not None:
            bias = bias.reshape(self.num_kv_heads, rep, -1).mean(dim=1).reshape(-1)
    elif weight.shape[0] < self.kv_head_size and self.kv_head_size % weight.shape[0] == 0:
        rep = self.kv_head_size // weight.shape[0]
        weight = weight.repeat(rep, 1)
        if bias is not None:
            bias = bias.repeat(rep)
    return weight, bias
```

## 5.4 SVD Factorization

Each projection is factorized via:

```python
def _factorize_and_set(self, target_linear, weight, bias=None):
    U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
    r = min(target_linear.rank, U.size(1), Vt.size(0))
    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-5))
    U_r = U[:, :r] * sqrt_S.unsqueeze(0)
    V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]
    target_linear.set_low_rank_factors(U_r, V_r)
    if bias is not None:
        target_linear.bias.data.copy_(bias)
```

## 5.5 Nested Encoder Handling

For RoBERTa/BERT, `is_nested_encoder = True`:
- The output projection is skipped (applied externally).
- The forward returns the raw context.

```python
if self.is_nested_encoder:
    output = context
else:
    output = self.out_proj(context)
```

## 5.6 Injection

```python
def inject_svd_linear_attention(layer, model_config=None, old_attn=None, **kwargs):
    target = layer
    cfg = model_config or target.config
    
    # Nested encoder detection
    is_nested = hasattr(target, "attention") and hasattr(target.attention, "self")
    if is_nested and target.attention is old_attn:
        old_attn = target.attention.self
    
    new_attn = SVDLinearAttention(config=cfg, old_attn=old_attn, **kwargs)
    new_attn.is_nested_encoder = is_nested
    
    if is_nested:
        target.attention.self = new_attn
    elif hasattr(target, "attention"):
        target.attention = new_attn
    elif hasattr(target, "self_attn"):
        target.self_attn = new_attn
    elif hasattr(target, "attn"):
        target.attn = new_attn
    
    return new_attn
```

## 5.7 The `step_recurrent` API

For streaming inference:

```python
def step_recurrent(self, x_t, past_state=None):
    """Process one token.
    
    Args:
        x_t: [B, d] or [B, 1, d]
        past_state: (S, Z) from previous step, or None for first step
    
    Returns:
        out_t: [B, 1, d]
        new_state: (S, Z)
    """
    # 1. Project, apply feature map
    # 2. Compute decay λ_t
    # 3. Update S, Z
    # 4. Compute output
    # 5. Apply out_proj
    return out_t, (S_t, Z_t)
```

**Usage example:**

```python
state = None
for token in sequence:
    out, state = attention.step_recurrent(token.unsqueeze(0), state)
    # out is the attention output for this token
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing projection | `copy_weights` raises | Injection aborts |
| Small Frobenius norm | Explicit check | Raises RuntimeError |
| GQA shape mismatch | Explicit reshape | Adapts |
| OOM | RuntimeError | Reduce batch |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("svd_linear_attn")
@ATTENTION_REGISTRY.register("svd_linear_attention")
@ATTENTION_REGISTRY.register("svd_linear")
class SVDLinearAttention(BaseAttention):
    ...
```

## 5.10 Causality

**Verification.** Gradient of output at position $i$ w.r.t. input at 
position $j > i$ is zero for the causal mode.

---