<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

| Architecture | Attention | Nested? | RoPE? | GQA? | Path |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | Bidirectional |
| RoBERTa | `BertSelfAttention` | Yes | No | No | Bidirectional |
| GPT-2 | `GPT2Attention` | No | No | No | Causal |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | Causal |

## 5.2 Weight Transfer

### 5.2.1 GPT-2 (Fused Conv1D)

```python
if hasattr(old_attn, "c_attn"):
    c_w = old_attn.c_attn.weight.data
    if "Conv1D" in cls_c or hasattr(old_attn.c_attn, "nf"):
        c_w = c_w.t()
    split_size = c_w.size(0) // 3
    extracted["q"] = c_w[:split_size, :]
    extracted["k"] = c_w[split_size:2*split_size, :]
    extracted["v"] = c_w[2*split_size:, :]
```

### 5.2.2 RoBERTa (Nested)

```python
elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
    extracted["q"] = old_attn.self.query.weight.data
    extracted["k"] = old_attn.self.key.weight.data
    extracted["v"] = old_attn.self.value.weight.data
    extracted["o"] = old_attn.output.dense.weight.data
```

### 5.2.3 DistilBERT (Flat)

```python
else:
    extracted["q"] = old_attn.q_lin.weight.data
    extracted["k"] = old_attn.k_lin.weight.data
    extracted["v"] = old_attn.v_lin.weight.data
    extracted["o"] = old_attn.out_lin.weight.data
```

### 5.2.4 The Z Projection

The **Z projection** is new. We initialize from $W_Q$:

```python
self.z_proj.weight.data[:out_z, :in_z].copy_(w_q[:out_z, :in_z])
```

## 5.3 Nested Encoder Detection

The detection order:

```python
if (hasattr(old_attn, "self") and hasattr(old_attn.self, "query")
    and hasattr(old_attn, "output") and hasattr(old_attn.output, "dense")):
    is_nested = True
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention", ...):
    is_nested = True
elif model_type in ("roberta", "bert", "xlm-roberta") and not is_decoder:
    is_nested = True
```

For nested encoders, `forward` returns raw context without `out_proj`.

## 5.4 GQA Support

```python
if self.num_kv_heads != self.num_heads:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

## 5.5 RoPE Support

Handled via partial-dimension slicing:

```python
rotary_dim = cos.shape[-1]
if rotary_dim < head_dim:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rot_embed = q_rot * cos + rotate_half(q_rot) * sin
    q = torch.cat([q_rot_embed, q_pass], dim=-1)
```

## 5.6 Injection Algorithm

**Algorithm 2:** Inject Gated DeltaNet into a model
```
Input:  model M, config c
Output: M with all attention replaced

1: layer_list ← find_transformer_layers(M)
2: for ℓ in layer_list:
3:     src ← extract_attention(ℓ)
4:     new ← GatedDeltaNet(c, old=src)
5:     new.copy_weights(src)
6:     replace_attention(ℓ, new)
7: return M
```

## 5.7 The Chunked Recurrence

The core implementation is `chunked_delta_rule_scan`:

```python
def chunked_delta_rule_scan(q, k, v, beta, alpha, chunk_size=8):
    # Normalize
    k_unit = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    q_unit = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
    # Reshape into chunks
    q_c = q_unit.view(B, H, num_chunks, chunk_size, d)
    ...
    
    S = torch.zeros(B, H, d, d)  # state
    outputs = []
    
    for c in range(num_chunks):       # ← OUTER PYTHON LOOP
        q_i, k_i, v_i = q_c[:,:,c], k_c[:,:,c], v_c[:,:,c]
        if use_checkpoint:
            y_i, S = checkpoint(_delta_chunk_forward, S, q_i, k_i, v_i, ...)
        else:
            y_i, S = _delta_chunk_forward(S, q_i, k_i, v_i, ...)
        outputs.append(y_i)
    
    return torch.cat(outputs, dim=2)

def _delta_chunk_forward(S, q_i, k_i, v_i, beta_i, alpha_i):
    for t in range(chunk_size):       # ← INNER PYTHON LOOP
        kt, vt, qt = k_i[:,:,t:t+1], v_i[:,:,t:t+1], q_i[:,:,t:t+1]
        bt, at = beta_i[:,:,t:t+1], alpha_i[:,:,t:t+1]
        
        inter_v = torch.matmul(kt, S)
        err = vt - inter_v
        norm_k_sq = (kt * kt).sum(-1, keepdim=True)
        step = (at / (1 + norm_k_sq).clamp(min=1.0)).clamp(0, 1)
        
        S = bt * S + step * torch.matmul(kt.transpose(-1, -2), err)
        yt = torch.matmul(qt, S)
    
    return torch.cat(outputs, dim=2), S
```

**Total Python iterations per forward:**
$$
L/C \text{ (outer)} + L/C \times C \text{ (inner)} = 2L/C + L \approx 2L
$$

For $L = 128$: **256 Python-level iterations per head per layer per forward**.

## 5.8 Common Pitfalls

### 5.8.1 Python Loop Overhead

Each `for` iteration involves:
- Tensor slicing (allocates new tensor)
- Kernel launch (~5-10 μs)
- Python dispatch (~1-2 μs)

For 256 iterations × 5 ops = ~1280 kernel launches per forward.

### 5.8.2 Gradient Checkpointing Overhead

`torch.utils.checkpoint` re-runs the forward during backward. This **doubles** 
compute cost in exchange for reduced memory.

### 5.8.3 State Memory

The state $S \in \mathbb{R}^{B \times H \times d \times d}$ is held in memory 
throughout. For $B=8$, $H=12$, $d=64$: 393K floats = **1.5 MB per layer**. 
With gradient checkpointing, states are re-created in backward, increasing 
peak memory.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("gated_delta_net")
class GatedDeltaNet(BaseAttention):
    ...
```

## 5.10 Recommended Fix

Replace Python loops with a **fused Triton or CUDA kernel**:

- Outer chunk loop: parallelize over chunks in a CUDA grid.
- Inner token loop: use shared memory for state, unroll with `#pragma unroll`.

Expected speedup: **50-100×**, bringing Gated DeltaNet to parity with 
baseline attention.

---