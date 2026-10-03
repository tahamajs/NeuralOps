<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

| Architecture | Attention | Nested? | RoPE? | GQA? | Path |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | Encoder |
| RoBERTa | `BertSelfAttention` | Yes | No | No | Encoder |
| GPT-2 | `GPT2Attention` | No | No | No | Decoder |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | Decoder |

## 5.2 Weight Transfer Details

### 5.2.1 Detecting Nested Encoders

The detection order matters. We check `has_out` first:

```python
has_out = any(hasattr(src, x) for x in ["dense", "out_proj", "out_lin", "c_proj", "o_proj"])
if cls_name in ["BertSelfAttention", "RobertaSelfAttention"] or (
    hasattr(src, "query") and not has_out
):
    self.is_nested_encoder = True
if hasattr(src, "self") and hasattr(src, "output"):
    self.is_nested_encoder = True
```

**Key insight.** DistilBERT's `MultiHeadSelfAttention` ends with "SelfAttention" but has `out_lin`, so `has_out=True` and it is **not** classified as nested. This is correct.

### 5.2.2 The RoPE Extraction Chain

For Qwen:

```python
if hasattr(src, "rotary_emb"):
    self.rotary_emb = src.rotary_emb
```

Then at forward time, we try three call signatures:

```python
try:
    cos, sin = rotary_mod(Q, seq_len=kv_len)
except Exception:
    try:
        cos, sin = rotary_mod(Q, pos_ids)
    except Exception:
        try:
            cos, sin = rotary_mod(Q)
        except Exception:
            cos, sin = None, None
```

**Fragile.** The triple-nested try/except works for common cases but fails 
silently for unusual RoPE implementations.

## 5.3 The Two Attention Paths

### 5.3.1 Encoder Path (Bidirectional)

```python
KV_cos = torch.matmul(K_cos.transpose(-2, -1), V)
KV_sin = torch.matmul(K_sin.transpose(-2, -1), V)
Z_cos = K_cos.sum(dim=2, keepdim=True)
Z_sin = K_sin.sum(dim=2, keepdim=True)

num = torch.matmul(Q_cos, KV_cos) + torch.matmul(Q_sin, KV_sin)
denom = (
    torch.matmul(Q_cos, Z_cos.transpose(-2, -1))
    + torch.matmul(Q_sin, Z_sin.transpose(-2, -1))
    + self.eps
)
context = num / denom.clamp(min=self.eps)
```

**4 matmuls, 2 sums.** Clean and fast.

### 5.3.2 Decoder Path (Chunked)

```python
chunk_size = 32
if N <= chunk_size:
    # Single chunk — direct computation with causal mask
    ...
else:
    # Multiple chunks
    for c in range(num_chunks):
        num_inter = Q_c @ S_cos + Q_sin @ S_sin
        denom_inter = Q_c @ Z_cos + Q_sin @ Z_sin
        # ... append, update state
```

**4 outer iterations for $L = 128$.** Each iteration ~5 kernel launches. 
Total ~20 extra kernel launches. Fast enough.

## 5.4 Masking

Masks are applied to **both K and V** (not just K):

```python
K = K * mask_expanded
V = V * mask_expanded
```

This is **critical** for the chunked scan — otherwise padding tokens 
contribute to the state.

## 5.5 Injection

**Algorithm 2:** Inject CosformerAttention
```
Input:  layer ℓ, config, old_attn
Output: new attention in ℓ

1: new ← CosformerAttention(config, old=old_attn)
2: if copy_weights fails: return None
3: if ℓ has .attention.self:
4:     ℓ.attention.self ← new; is_nested ← True
5: elif ℓ has .attention:
6:     ℓ.attention ← new
7: elif ℓ has .self: ℓ.self ← new; is_nested ← True
8: elif ℓ has .attn: ℓ.attn ← new
9: return new
```

## 5.6 Common Pitfalls

### 5.6.1 RoPE Call Signature

The triple-nested try/except for RoPE is fragile. If the RoPE implementation 
uses an unusual call signature, extraction fails silently.

### 5.6.2 Nested Encoder Detection

Getting `is_nested_encoder` wrong leads to double `out_proj`.

### 5.6.3 The `tau` Parameter is Unused

`self.tau = nn.Parameter(torch.tensor(1.0), requires_grad=False)` is 
declared but never used in `forward`. Dead code.

### 5.6.4 Padding in the Chunked Scan

Padding tokens must be masked in the state update. Otherwise, the state 
accumulates garbage.

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("cosformer")
@ATTENTION_REGISTRY.register("cosformer_attention")
class CosformerAttention(BaseAttention): ...
```

## 5.8 Integration with LoRA Recovery

The standard pipeline applies:

1. Load pretrained model.
2. Inject Cosformer via `inject_cosformer`.
3. Freeze base weights, keep LoRA trainable.
4. Run 50 recovery steps.
5. Evaluate.

## 5.9 Architecture-Specific Notes

### 5.9.1 DistilBERT

- **Path:** Encoder
- **Nested:** No
- **RoPE:** No

### 5.9.2 RoBERTa

- **Path:** Encoder
- **Nested:** Yes (output projection via outer `output.dense`)
- **RoPE:** No
- **Complication:** RoBERTa's `BertAttention` wraps `self` and `output`. We 
  replace `self`. The outer `output.dense` applies the projection.

### 5.9.3 GPT-2

- **Path:** Decoder
- **Nested:** No
- **RoPE:** No
- **Complication:** GPT-2 uses Conv1D weight layout (transposed).

### 5.9.4 Qwen3.5

- **Path:** Decoder
- **Nested:** No
- **RoPE:** Yes
- **GQA:** Yes
- **Complication:** RoPE extraction from HuggingFace API is fragile.

---