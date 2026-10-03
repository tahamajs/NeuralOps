# 4. Method

## 4.1 Overview

LoRA Attention replaces each of the four attention projections with a 
**LoRALinear** module:

$$
y = \underbrace{W_\text{frozen} x}_{\text{frozen, copied from pretrained}} + 
    \underbrace{\frac{\alpha}{r_L} B A x}_{\text{trainable LoRA}}
\tag{4.1}
$$

**Figure 4.1** shows the pipeline. The base weight is copied from the 
pretrained attention via `copy_weights`, and LoRA adapters are initialized 
to zero ($B = 0$), so the initial forward pass exactly matches the baseline.

```
        Input X
           │
           ▼
    ┌──────────────────────┐
    │ Base W (frozen)      │
    │ + LoRA (trainable)   │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Q, K, V projections  │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ RoPE (if applicable) │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ GQA repeat (if app.) │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Scaled dot-product   │
    │ attention            │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Output projection    │
    │ + LoRA               │
    └───────┬──────────────┘
            │
        Output
```

## 4.2 LoRALinear Module

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 lora_rank=16, lora_alpha=32.0, freeze_base=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            # A: Kaiming, B: zeros
            nn.init.kaiming_uniform_(self.lora_A)
            nn.init.zeros_(self.lora_B)
        
        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
    
    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

**Note.** We add a trainable `output_scale` scalar initialized to 1.0. This 
allows the model to globally re-scale the attention output if needed.

## 4.3 Weight Transfer

`copy_weights` extracts base weights from the pretrained attention and 
copies them into the LoRALinear modules. It handles four patterns:

### 4.3.1 Pattern 1: GPT-2 Conv1D

```python
if hasattr(src, "c_attn"):
    weight = src.c_attn.weight.data
    if weight.shape[1] == 3 * self.hidden_size:
        weight = weight.t()
    q_w, k_w, v_w = weight.chunk(3, dim=0)
    extracted["q"] = (q_w, ...)
    extracted["k"] = (k_w, ...)
    extracted["v"] = (v_w, ...)
```

### 4.3.2 Pattern 2: RoBERTa Nested

```python
if hasattr(src, "self"):
    s = src.self
    extracted["q"] = (s.query.weight, ...)
    extracted["k"] = (s.key.weight, ...)
    extracted["v"] = (s.value.weight, ...)
if hasattr(src, "output") and hasattr(src.output, "dense"):
    extracted["o"] = (src.output.dense.weight, ...)
```

### 4.3.3 Pattern 3: Named Projections

```python
named_map = {
    "q": ["q_proj", "q_lin", "query"],
    "k": ["k_proj", "k_lin", "key"],
    "v": ["v_proj", "v_lin", "value"],
    "o": ["o_proj", "out_proj", "out_lin", "dense"],
}
```

### 4.3.4 Pattern 4: UniversalAdapter Fallback

```python
for proj in ["q", "k", "v", "o"]:
    try:
        w, b = UniversalAdapter.get_weight_bias(src, proj)
        if w is not None:
            extracted[proj] = (w, b)
    except Exception:
        pass
```

## 4.4 Nested Detection

The critical detection logic:

```python
cls_name = src.__class__.__name__
has_out_proj = any(hasattr(src, attr) for attr in 
                   ("dense", "out_proj", "c_proj", "out_lin", "o_proj", ...))

is_nested = False
if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
    is_nested = True
elif (cls_name in ("BertSelfAttention", "RobertaSelfAttention", ...)
      or (cls_name.endswith("SelfAttention") and not has_out_proj)) and not has_out_proj:
    is_nested = True
self.is_nested_encoder = bool(is_nested)
```

**Rule:**
- If `self.query` exists → nested (RoBERTa).
- If class name is `BertSelfAttention`/`RobertaSelfAttention` → nested.
- If class name ends with "SelfAttention" **AND** lacks any output projection → nested.
- Otherwise → flat.

## 4.5 Forward Pass

**Algorithm 1:** LoRA Attention forward
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y ∈ R^{B×L×d}

 1: Q ← q_proj(X)                       ▷ base + LoRA
 2: K ← k_proj(X)                       ▷ base + LoRA
 3: V ← v_proj(X)                       ▷ base + LoRA
 4: Reshape to (B, H, L, d_h)
 5: if RoPE: Q, K ← apply_rotary(Q, K)
 6: if GQA: K, V ← repeat_kv(K, V)
 7: scores ← Q K^T / √d_h
 8: if causal: mask(scores, tril)
 9: if padding: mask(scores, valid)
10: attn ← softmax(scores)
11: Context ← attn V
12: Context ← reshape(B, L, H d_h)
13: if is_nested_encoder:
14:     return Context  ▷ outer layer applies out_proj
15: Y ← out_proj(Context)
16: return Y
```

## 4.6 Parameter Analysis

Per attention layer (excluding base weights):

| Component | Params |
|---|---|
| `lora_A_q` | $r_L \cdot d$ |
| `lora_B_q` | $H d_h \cdot r_L$ |
| `lora_A_k` | $r_L \cdot d$ |
| `lora_B_k` | $H_{kv} d_h \cdot r_L$ |
| `lora_A_v` | $r_L \cdot d$ |
| `lora_B_v` | $H_{kv} d_h \cdot r_L$ |
| `lora_A_o` | $r_L \cdot H d_h$ |
| `lora_B_o` | $d \cdot r_L$ |
| `output_scale` | 1 |
| **Total** | $r_L (2d + 2 H d_h)$ |

For $d = 768$, $d_h = 64$, $H = 12$, $r_L = 16$:

$$
\text{Trainable}_\text{LoRA} = 16 \cdot (2 \cdot 768 + 2 \cdot 768) = 49,152
\tag{4.2}
$$

Per layer: ~49K trainable, compared to 2.36M frozen parameters.

## 4.7 Mergeability

After training, we can merge LoRA into the base weight:

$$
W_\text{merged} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{4.3}
$$

Then:

$$
y = W_\text{merged} x + b
\tag{4.4}
$$

**Zero inference overhead** — the merged model is identical in structure to 
the original.

## 4.8 Complexity Analysis

### 4.8.1 Training Complexity

| Operation | Full FT | LoRA Attention |
|---|---|---|
| Forward | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2 d_h)$ |
| Backward (base) | $\mathcal{O}(L^2 d_h)$ | 0 |
| Backward (LoRA) | — | $\mathcal{O}(L^2 d_h \cdot r_L/d)$ |
| Optimizer step | $\mathcal{O}(d^2)$ | $\mathcal{O}(r_L d)$ |
| Memory (per param) | 10 bytes | 10 bytes (only adapters) |

### 4.8.2 Inference Complexity

After merging: **identical to baseline**.

Before merging: adds $\mathcal{O}(L \cdot r_L \cdot d)$ per projection.

## 4.9 Theoretical Notes

### 4.9.1 Zero Initialization

**Claim.** At initialization, LoRA Attention is exactly equivalent to the 
pretrained attention.

**Proof.** $B = 0$, so $\Delta W = 0$. Thus $W_\text{eff} = W_\text{frozen}$. $\square$

### 4.9.2 Mergeability Preserves Expressivity

**Claim.** Any parameter set achievable by the LoRA branch is achievable 
by the merged weight.

**Proof.** $W_\text{merged} = W_\text{frozen} + BA$. Trivially achievable. $\square$

### 4.9.3 Nested Encoder Correctness

**Claim.** For nested encoders, skipping the output projection yields 
correct attention output.

**Proof.** The nested outer layer applies $W_O$. Our module returning raw 
context matches the original architecture's flow. $\square$

---