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


## 5.1 Model Support

A3 supports any transformer with linear or Conv1D layers:

| Architecture | Attention | FFN | Conv1D? | Verified |
|---|---|---|---|---|
| DistilBERT-base | `q_lin, k_lin, v_lin, out_lin` | `lin1, lin2` | No | ✅ |
| RoBERTa-base | `query, key, value, output.dense` | `intermediate.dense, output.dense` | No | ✅ |
| GPT-2 | `c_attn` (fused), `c_proj` | `c_fc, c_proj` | Yes | ✅ |
| Qwen3.5-0.8B | `q_proj, k_proj, v_proj, o_proj` | `gate_proj, up_proj, down_proj` | No | ✅ |

## 5.2 Layer Targeting

A3 compresses all linear layers by default, with the following protections:

| Layer type | Compressed? | Reason |
|---|---|---|
| `classifier`, `score`, `lm_head` | ❌ Never | Small and critical |
| `pooler`, `embedding` | ❌ Never | Not linear |
| Attention projections | ✅ Yes | Target of interest |
| FFN layers | ✅ Yes (if `target_attention_only = False`) | Large parameter share |
| LayerNorm | ❌ Never | Not factorizable |

The `should_compress_linear` helper enforces these rules.

## 5.3 Handling Different Architectures

### 5.3.1 DistilBERT (Flat)

Standard linear layers:

```python
distilbert.transformer.layer.0.attention.q_lin  # Linear(768, 768)
distilbert.transformer.layer.0.attention.k_lin  # Linear(768, 768)
distilbert.transformer.layer.0.attention.v_lin  # Linear(768, 768)
distilbert.transformer.layer.0.attention.out_lin # Linear(768, 768)
distilbert.transformer.layer.0.ffn.lin1          # Linear(768, 3072)
distilbert.transformer.layer.0.ffn.lin2          # Linear(3072, 768)
```

**A3 compresses each directly.** No special handling needed.

### 5.3.2 RoBERTa (Nested)

Standard linear layers within the nested `BertAttention`:

```python
roberta.encoder.layer.0.attention.self.query  # Linear(768, 768)
roberta.encoder.layer.0.attention.self.key    # Linear(768, 768)
roberta.encoder.layer.0.attention.self.value  # Linear(768, 768)
roberta.encoder.layer.0.attention.output.dense # Linear(768, 768)
roberta.encoder.layer.0.intermediate.dense    # Linear(768, 3072)
roberta.encoder.layer.0.output.dense          # Linear(3072, 768)
```

**All are standard Linear.** No special handling.

### 5.3.3 GPT-2 (Conv1D)

GPT-2 uses `Conv1D` from HuggingFace, which stores weights as 
`(in_features, out_features)` instead of `(out_features, in_features)`:

```python
transformer.h.0.attn.c_attn  # Conv1D(768, 2304) — fused Q,K,V
transformer.h.0.attn.c_proj  # Conv1D(768, 768)
transformer.h.0.mlp.c_fc     # Conv1D(768, 3072)
transformer.h.0.mlp.c_proj   # Conv1D(3072, 768)
```

**A3 handles Conv1D** by transposing the weight before SVD:

```python
if "conv1d" in cls_name or hasattr(module, "nf"):
    W = raw_w.t()  # (in, out) → (out, in)
```

The factorized layer replaces the Conv1D. Because the factorized layer 
implements a standard linear operation, the `nf`/`nx` attributes are preserved 
for compatibility.

### 5.3.4 Qwen3.5 (GQA + RoPE)

Standard PyTorch Linear layers with GQA structure:

```python
model.layers.0.self_attn.q_proj  # Linear(1024, 2048) — H * d_h
model.layers.0.self_attn.k_proj  # Linear(1024, 512)  — H_kv * d_h
model.layers.0.self_attn.v_proj  # Linear(1024, 512)
model.layers.0.self_attn.o_proj  # Linear(2048, 1024)
```

**A3 compresses each directly.** GQA-specific handling is unnecessary because 
the factorization is per-projection.

## 5.4 Wrapper Compatibility

A3 supports models wrapped with LoRA-style adapters (`LoRALinear`, etc.):

```python
class LoRALinear(nn.Module):
    def __init__(...):
        self.base_layer = nn.Linear(...)  # ← A3 targets this
        self.lora_A = nn.Parameter(...)
        self.lora_B = nn.Parameter(...)
```

**Injection procedure:**

1. Extract `target_mod = getattr(module, "base_layer", module)`.
2. Factorize `target_mod`.
3. Replace `module.base_layer` with the factorized layer.
4. **Preserve** the outer LoRA parameters (`module.lora_A`, `module.lora_B`).

If the original module had LoRA, A3 optionally instantiates LoRA inside the 
factorized layer to maintain trainability. This is controlled by 
`lora_rank > 0`.

## 5.5 Injection Algorithm

**Algorithm 2:** Inject A3 into a model
```
Input:  model M, config c, calibration data D
Output: compressed model M'

1: hooks ← []
2: for ℓ in linear_modules(M) do
3:     hooks.append(register_forward_hook(ℓ, accumulate_F_C))
4: end for
5: run_calibration(M, D)
6: for h in hooks: h.remove()
7:
8: F_map ← finalized_statistics()
9: for ℓ in linear_modules(M) do
10:     if not should_compress(ℓ): continue
11:     U, V, err ← weighted_svd(ℓ.weight, F_map[ℓ], r, τ)
12:     if err ≤ τ:
13:         replace(ℓ, A3FactorizedLinear(U, V))
14:     else:
15:         mark_fallback(ℓ)
16: end for
17:
18: if recalibrate_ln:
19:     σ_target ← record_LN_before(M, D)  ▷ pre-computed
20:     σ_current ← record_LN_after(M', D)
21:     for γ in layer_norms(M') do
22:         γ ← γ · σ_target / σ_current
23:     end for
24: end if
25: return M'
```

## 5.6 The `A3FactorizedLinear` Module

The factorized layer implements:

```python
class A3FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, ...):
        self.V = nn.Parameter(torch.empty(rank, in_features))  # (r, d_in)
        self.U = nn.Parameter(torch.empty(out_features, rank)) # (d_out, r)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        # Optional LoRA
        if lora_rank > 0:
            self.lora_A = nn.Parameter(...)
            self.lora_B = nn.Parameter(...)
    
    def forward(self, x):
        latent = F.linear(x, self.V)      # (B, L, r)
        out = F.linear(latent, self.U)    # (B, L, d_out)
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        if self.bias is not None:
            out += self.bias
        return out
```

**Parameter count:** `r * (d_in + d_out)` vs `d_in * d_out` for the original.

## 5.7 Causality and Streaming

For decoder models, A3's calibration is causal-compatible:

- Forward hooks capture activations as they occur
- No future-token leakage because hooks run during the standard causal forward pass

**Verification.** We checked that gradient of output at position $i$ w.r.t. 
input at position $j > i$ is zero, both before and after injection.

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Reconstruction error > τ | Frobenius check | Fallback to original |
| Weight shape mismatch | Assertion in `from_linear` | Skip layer |
| SVD convergence failure | Try/except | Retry on CPU |
| No calibration data | `F_map` empty | Unweighted SVD |
| CUDA OOM in SVD | RuntimeError | Retry on CPU |
| NaN in reconstruction | `torch.isnan` | Skip and log |

## 5.9 Registry

```python
@COMPRESSOR_REGISTRY.register("a3")
@COMPRESSOR_REGISTRY.register("a3_compressor")
def a3_compressor_factory(model, comp_config, ...):
    ...
```

## 5.10 Compatibility with LoRA Recovery Pipeline

A3 integrates cleanly with the LoRA recovery pipeline:

1. **A3 compresses** the base model → factorized layers.
2. **LoRA adapters** are attached to the factorized layers.
3. **Recovery fine-tuning** trains only the LoRA parameters.
4. **Evaluation** uses the fine-tuned model.

This is the same pipeline used by our framework's other compression methods.
