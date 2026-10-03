<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

SVD-LLM V2 replaces each attention projection with **three explicit 
buffers**:

$$
W \to (U, \Sigma, V^H)
\tag{4.1}
$$

The forward pass computes:

$$
y = ((x V^H{}^\top) \Sigma) U^\top + b + \text{LoRA}(x)
\tag{4.2}
$$

Rank is chosen per projection by energy threshold. LoRA adapters provide 
recovery.

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────┐
   │  Input weight W ∈ R^{d_out × d_in}           │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  SVD: W = U Σ V^H                             │
   │  Compute cumulative energy E(r)              │
   │  r_eff = min(r_max, argmin E(r) ≥ η)         │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Store explicitly:                            │
   │  U  ∈ R^{d_out × r}                           │
   │  Σ  ∈ R^r (diagonal)                          │
   │  V^H ∈ R^{r × d_in}                           │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Freeze U, Σ, V^H. Add LoRA (A, B=0)          │
   └──────────────────────────────────────────────┘
```

## 4.2 The Explicit Factor Buffers

For each projection (Q, K, V, out), we allocate:

| Buffer | Shape | Purpose |
|---|---|---|
| `*_u` | $(d_\text{out}, r)$ | Left singular vectors |
| `*_s` | $(r,)$ | Singular values (1D) |
| `*_vh` | $(r, d_\text{in})$ | Right singular vectors |
| `*_bias` | $(d_\text{out},)$ | Bias (separate) |

Where `*` is one of `query`, `key`, `value`, `out_proj`.

**Memory savings.** For $d = 768$ and $r = 128$:

| Config | Params |
|---|---|
| Original | 589,824 |
| Explicit (U + Σ + V^H) | $768 \times 128 + 128 + 128 \times 768 = 196{,}736$ |
| Ratio | **3.0×** |

With rank 64: **6.0×** reduction.

## 4.3 The `_project` Helper

SVD-LLM V2 uses a unified helper:

```python
def _project(self, x, vh, s, u, b, lora_A, lora_B):
    base_out = ((x @ vh.t()) * s.to(dtype=x.dtype)) @ u.t() + b
    if self.lora_rank > 0 and lora_A is not None and lora_B is not None:
        lora_out = F.linear(F.linear(x, lora_A), lora_B) * self.scaling
        base_out = base_out + lora_out
    return base_out
```

**Order of operations:**

1. Down-project: $x V^H{}^\top$ → $(B, L, r)$
2. Scale: multiply by $\Sigma$ → $(B, L, r)$
3. Up-project: $\cdot U^\top$ → $(B, L, d_\text{out})$
4. Add bias
5. Add LoRA residual

## 4.4 Energy-Threshold Rank Selection

Given $W$, target rank cap $r_\text{max}$, and energy $\eta$:

1. Compute SVD: $W = U \Sigma V^H$.
2. Compute cumulative energy: $E(r) = \sum_{i \leq r} \sigma_i^2 / \sum_i \sigma_i^2$.
3. Find: $r_\text{energy} = |\{i : E(i) < \eta\}| + 1$.
4. Cap: $r_\text{eff} = \min(r_\text{energy}, r_\text{max}, \min(d_\text{in}, d_\text{out}))$.

The rank is set **per projection** (not per layer). This means Q, K, V, out 
may each get a different rank.

## 4.5 Weight Transfer

### 4.5.1 Architecture Patterns

| Architecture | Extraction |
|---|---|
| GPT-2 | Split `c_attn` into 3 chunks; separate `c_proj` |
| RoBERTa | Extract `self.query/key/value` and `output.dense` |
| Qwen | Extract `q_proj/k_proj/v_proj` and `o_proj` |
| DistilBERT | Extract `q_lin/k_lin/v_lin` and `out_lin` |
| Fallback | UniversalAdapter |

### 4.5.2 Conv1D Handling

```python
def _unpack_weight_bias(module):
    base = getattr(module, "base_layer", module)
    w = base.weight.data.clone()
    if "Conv1D" in type(module).__name__ or hasattr(module, "nf"):
        w = w.t()   # (in, out) → (out, in)
    return w, base.bias
```

### 4.5.3 GQA Pooling

When `w_k.shape[0] != self.kv_head_size`:

```python
rep = w_k.shape[0] // self.kv_head_size
w_k_pooled = (
    w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
    .mean(dim=1)
    .view(self.kv_head_size, -1)
)
```

Same for bias.

## 4.6 Nested Encoder Detection

`is_nested_encoder` is set based on the presence of `src.self.query`:

```python
if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
    # Nested encoder — extract from inner, skip out_proj
    w_o = src.output.dense.weight
    self.is_nested_encoder = True
```

When nested, `out_proj` is skipped in forward.

## 4.7 Injection Protocol

**Algorithm 1:** Whole-model injection with pre-test
```
Input:  model M, rank cap r_max, energy η
Output: M' with factorized attention

1:  layer_list ← find_transformer_layers(M)
2:
3:  # Pre-test on first layer
4:  test_src ← get_attention(layer_list[0])
5:  try:
6:      test_module ← SVDLLMV2Attention(cfg, old_attn=test_src)
7:  except Exception:
8:      return None
9:
10: # Inject each layer
11: count ← 0
12: for (idx, layer) in enumerate(layer_list) do
13:     src_attn ← get_attention(layer)
14:     if src_attn is SVDLLMV2Attention: skip
15:     new_attn ← SVDLLMV2Attention(cfg, old_attn=src_attn)
16:
17:     # Set nested flag and replace
18:     for attr in ("attention", "attn", "self_attn", "self"):
19:         if hasattr(layer, attr):
20:             if attr == "attention" and hasattr(layer.attention, "self"):
21:                 new_attn.is_nested_encoder = True
22:                 layer.attention.self = new_attn
23:             else:
24:                 new_attn.is_nested_encoder = False
25:                 setattr(layer, attr, new_attn)
26:             break
27:     count += 1
28: end for
29:
30: return M' if count > 0 else None
```

## 4.8 LoRA Recovery

Each projection has its own LoRA pair:

- `q_lora_A`, `q_lora_B`
- `k_lora_A`, `k_lora_B`
- `v_lora_A`, `v_lora_B`
- `out_lora_A`, `out_lora_B`

All `*_B` are initialized to zero. Base weights ($U$, $\Sigma$, $V^H$, biases) 
are frozen.

## 4.9 Complexity Analysis

| Phase | Time | Memory |
|---|---|---|
| Extraction | $\mathcal{O}(N_l d^2)$ | $\mathcal{O}(d^2)$ |
| SVD per projection | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^2)$ |
| Truncation | $\mathcal{O}(N_l r d^2)$ | $\mathcal{O}(d^2)$ |
| Total | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |

## 4.10 Theoretical Notes

### 4.10.1 Energy Bound

**Proposition.** If $r_\text{eff}$ is the smallest rank with 
$E(r_\text{eff}) \geq \eta$, then:

$$
\frac{\|W - W_{r_\text{eff}}\|_F}{\|W\|_F} \leq \sqrt{1 - \eta}
\tag{4.3}
$$

**Proof.** $\|W - W_r\|_F^2 = \|W\|_F^2 (1 - E(r))$. $\square$

### 4.10.2 Explicit vs Symmetric Factorization

Both forms are mathematically equivalent:

$$
(U \Sigma^{1/2}) (\Sigma^{1/2} V^H) = U \Sigma V^H
\tag{4.4}
$$

The explicit form exposes $\Sigma$ for interpretation. The symmetric form is 
numerically stabler in FP16 because both factors have unit spectral norm.

**SVD-LLM V2 default:** explicit. For FP16-only deployments, consider 
switching to symmetric.

## 4.11 Summary

SVD-LLM V2:
1. Factorizes each attention projection into explicit $(U, \Sigma, V^H)$.
2. Selects rank per projection via energy threshold.
3. Uses LoRA for recovery.
4. Handles four attention architectures with pre-test and GQA pooling.
5. Achieves **real parameter reduction** of $d^2 / (r(d_\text{in} + d_\text{out} + 1))$.

---