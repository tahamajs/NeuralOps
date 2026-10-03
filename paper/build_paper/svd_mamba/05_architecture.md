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

### 5.2.4 Qwen

Direct extraction from `q_proj/k_proj/v_proj/o_proj`.

## 5.3 Hybrid Initialization

SVD-Mamba's `copy_weights` does three things:

1. **Attention path:** Factorize $W_Q, W_K, W_V, W_O$ into SVD factors 
   for the attention projections.
2. **SSM input projection:** Build a concatenated $[V; Q]$ weight to 
   initialize `in_proj`.
3. **Output projection:** Copy the attention out projection to `attn_out_proj`.

**The SSM branch is NOT initialized from the attention weights.** It starts 
with random weights, but its contribution is silenced by the gate.

## 5.4 The Gating Module

```python
self.gating = nn.Linear(self.d_model, self.d_model)
nn.init.zeros_(self.gating.weight)
nn.init.constant_(self.gating.bias, -7.0)
```

**Why bias $-7$?** The gating controls the SSM path:

- $\sigma(-7) \approx 0.0009$ → SSM contributes <0.1% at start.
- The model starts as pure attention.
- During recovery, the gate can shift toward SSM.

## 5.5 Nested Encoder Handling

For RoBERTa/BERT, `is_nested_encoder = True` means SVD-Mamba returns the 
raw output. The outer `attention.output.dense` applies the actual output 
projection.

But note: SVD-Mamba's forward sets `is_nested_encoder = True` in the 
constructor if the source has the right structure, and the forward 
returns `Y` directly. There's no explicit bypass code — the output 
shape is already `[B, L, d]`.

## 5.6 Injection

The injection hook handles:

- Whole-model traversal
- Single layer replacement
- Double-injection guard
- Qwen DeltaNet skip

```python
def inject_svd_mamba(layer=None, model=None, ...):
    target = layer or model
    
    # Whole model
    if isinstance(target, nn.Module) and has_transformer_layers(target):
        for lyr in layers:
            inject_svd_mamba(layer=lyr, ...)
        return target
    
    # Single layer
    src = find_attention(target)
    new_module = SVDMamba(d_model=..., pretrained_attn=src)
    replace_attention(target, new_module)
    return new_module
```

## 5.7 The Chunked SSM Scan

The forward uses a Python loop over chunks:

```python
chunk_size = 64
num_chunks = (seqlen + chunk_size - 1) // chunk_size
h_running = zeros(...)
h_den_running = zeros(...)
y_chunks = []

for c in range(num_chunks):
    s_idx = c * chunk_size
    e_idx = min(seqlen, s_idx + chunk_size)
    
    u_c = u[:, s_idx:e_idx]
    log_a_c = log_alpha[:, s_idx:e_idx]
    lambda_c = cumsum(log_a_c, dim=1).clamp(min=-25, max=0)
    
    decay_c = exp(lambda_c)
    decay_inv_c = exp(-lambda_c).clamp(max=1e10)
    
    h_intra = decay_c * cumsum(decay_inv_c * u_c, dim=1)
    h_inter = decay_c * h_running.unsqueeze(1)
    h_chunk = (h_intra + h_inter).clamp(min=-1e4, max=1e4)
    
    # ... denominator ...
    y_chunks.append(...)
    h_running = h_chunk[:, -1]
```

**Python loop overhead** is significant: for $L = 128$, 2 iterations; for 
$L = 1024$, 16 iterations. Each iteration has $\sim 10$ kernel launches.

## 5.8 Numerical Stability

Several clamps prevent overflow:

| Operation | Clamp |
|---|---|
| `log_alpha` cumsum | $[-25, 0]$ |
| `h_chunk` | $[-10^4, 10^4]$ |
| `h_den_chunk` | $[0, 10^4]$ |
| `decay_inv` | $\leq 10^{10}$ |
| `denom` | $\geq 10^{-4}$ |
| `A_log` | $[-20, 10]$ |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("svd_mamba")
@ATTENTION_REGISTRY.register("SVD-Mamba")
@ATTENTION_REGISTRY.register("svd-mamba")
@ATTENTION_REGISTRY.register("SVD_Mamba")
class SVDMamba(BaseAttention): ...
```

## 5.10 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing projection | `copy_weights` raises | Injection aborts |
| Small Frobenius norm | Explicit check | Raises RuntimeError |
| NaN in scan | `torch.nan_to_num` | Fallback |
| OOM | RuntimeError | Reduce batch |
| Cross-attention | Returns input | No-op |

## 5.11 Cross-Attention

If `encoder_hidden_states` is provided, SVD-Mamba returns 
`(hidden_states, None)` — a pass-through. This means cross-attention is 
effectively unsupported.

---