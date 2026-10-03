<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

| Architecture | Nested? | RoPE? | GQA? | Path |
|---|---|---|---|---|
| DistilBERT | No | No | No | Bidirectional |
| RoBERTa | Yes | No | No | Bidirectional |
| GPT-2 | No | No | No | Causal |
| Qwen3.5 | No | Yes | Yes | Causal |

## 5.2 Weight Transfer

Standard four-architecture copy.

### 5.2.1 GPT-2 Conv1D

```python
c_w = src.c_attn.weight.data
if is_conv1d:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa Nested

```python
q_w = src.self.query.weight.data
k_w = src.self.key.weight.data
v_w = src.self.value.weight.data
o_w = src.output.dense.weight.data
```

### 5.2.3 DistilBERT Flat

```python
q_w = src.q_lin.weight.data
# ...
```

## 5.3 Feature Map and Masking

**Encoder path** masks both K and Q:

```python
K_feat = self._feature_map(K) * mask_expanded
Q_feat = self._feature_map(Q) * mask_expanded  # ← Important
```

Masking Q ensures that padding queries do not contribute to the output.

**Decoder path** only masks K:

```python
K_feat = self._feature_map(K) * mask_expanded
# Q_feat is not masked
```

Because causal attention already limits the influence of future tokens.

## 5.4 Causal Chunked Scan

```python
chunk_size = min(64, N)
num_chunks = (N + chunk_size - 1) // chunk_size
```

For $N = 128$: 2 chunks.

**Intra-chunk attention:**

```python
causal_mask = torch.tril(torch.ones(L, L, ...))
intra_attn = torch.matmul(q_c, k_c.transpose(-1, -2)) * causal_mask
```

**Inter-chunk via running state:**

```python
inter_num = torch.matmul(q_c, S)
S = S + torch.matmul(k_c.transpose(-1, -2), v_c)
```

## 5.5 Injection

**Algorithm 2:** Inject Performer
```
1: src ← extract_attention(ℓ)
2: new ← PerformerAttention(config, old=src)
3: if copy_weights fails: raise (BUG — should return None)
4: install new in ℓ.attention.self (or .attn, .self_attn)
5: return new
```

## 5.6 Common Pitfalls

### 5.6.1 Injection Raises

The docstring claims `returning None on failure` but the code raises. 
Wrap the call in `try/except` if you need None-return behavior.

### 5.6.2 `output_scale` Redundancy

Each `LoRALinear` has its own `output_scale`. If you wrap Performer in 
another module that also scales, double-scaling can occur.

### 5.6.3 Cross-Attention Bypass

```python
if encoder_hidden_states is not None:
    return hidden_states, None
```

Cross-attention is **not supported**. The module returns the input 
unchanged.

### 5.6.4 The `omega` Buffer is Persistent

```python
self.register_buffer("omega", omega, persistent=True)
```

Good — the RFF projection is saved and reloaded.

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("performer")
@ATTENTION_REGISTRY.register("performer_attention")
class PerformerAttention(BaseAttention): ...
```

## 5.8 Integration with LoRA Recovery

Standard pipeline:
1. Load pretrained model.
2. Inject Performer via `inject_performer`.
3. Freeze base weights, keep LoRA trainable.
4. Run 50 recovery steps.
5. Evaluate.

---