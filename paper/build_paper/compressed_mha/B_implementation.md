# Appendix B. Implementation Details

## B.1 Core Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, **kwargs):
    B, N, _ = hidden_states.shape
    
    # 1. Joint QKV projection (SVD + LoRA)
    proj = torch.matmul(hidden_states, self.Vh_qkv.t()) * self.S_qkv.unsqueeze(0).unsqueeze(0)
    qkv_comp = torch.matmul(proj, self.U_qkv.t())
    if self.lora_rank > 0:
        lora_qkv = F.linear(F.linear(hidden_states, self.lora_A_qkv), self.lora_B_qkv) * self.lora_scaling
        qkv_comp = qkv_comp + lora_qkv
    
    # 2. Split
    q_raw, k_raw, v_raw = torch.split(qkv_comp, 
                                       [self.all_head_size, self.kv_head_size, self.kv_head_size], dim=-1)
    q = q_raw + self.bias_q
    k = k_raw + self.bias_k
    v = v_raw + self.bias_v
    
    # 3. Reshape
    q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
    
    # 4. RoPE
    if position_embeddings is not None:
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    
    # 5. GQA
    if self.kv_repeat > 1:
        k = repeat_kv(k, self.kv_repeat)
        v = repeat_kv(v, self.kv_repeat)
    
    # 6. Attention
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
    # masks
    if is_decoder:
        causal = torch.tril(torch.ones(N, N))
        scores = scores.masked_fill(~causal, -1e4)
    attn = F.softmax(scores.float(), dim=-1)
    context = torch.matmul(attn, v)
    context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
    
    # 7. Output
    if self.is_nested_encoder:
        return context, None
    out_proj = torch.matmul(context, self.Vh_out.t()) * self.S_out.unsqueeze(0).unsqueeze(0)
    out = torch.matmul(out_proj, self.U_out.t()) + self.bias_out
    if self.lora_rank > 0:
        lora_out = F.linear(F.linear(context, self.lora_A_out), self.lora_B_out) * self.lora_scaling
        out = out + lora_out
    return out, None
```

## B.2 Nested Detection Fix

```python
# Priority 1: explicit flat output projections
if hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj"):
    self.is_nested_encoder = False
# Priority 2: nested structure
elif (hasattr(src, "self") and hasattr(src, "output") 
      and hasattr(src.output, "dense")):
    self.is_nested_encoder = True
# Priority 3: class name (last resort)
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention"):
    self.is_nested_encoder = True
else:
    self.is_nested_encoder = False
```

**Why the order matters:** `DistilBert.MultiHeadSelfAttention` ends with 
"SelfAttention" — if we checked class name first, we'd incorrectly mark it 
as nested.

## B.3 Safe Weight Extraction

```python
for proj in ["q", "k", "v", "o"]:
    try:
        w, b = UniversalAdapter.get_weight_bias(src, proj)
        if w is not None:
            extracted[proj] = (w, b)
    except Exception as e:
        logger.debug(f"UniversalAdapter failed for {proj}: {e}")
        # Continue to next extraction method
```

**No `raise`** — failures are logged at debug level and extraction continues.

## B.4 Rank Defaults

```python
rank_val = rank or kwargs.get("rank_qkv", rank_qkv)
if rank is None and "rank_qkv" not in kwargs:
    rank_val = max(rank_qkv, min(384, all_head_size + 2 * kv_head_size, hidden_size))
self.rank_qkv = min(rank_val, all_head_size + 2 * kv_head_size, hidden_size)
```

**Important:** The default value is designed to be **safe but not compressive** 
(rank = 384 for $d=768$). For real compression, pass explicit ranks.

## B.5 Common Pitfalls

### B.5.1 Wrong Nested Detection Order

Checking class name before `out_lin` marks DistilBERT as nested.

### B.5.2 Forgetting RoPE

For Qwen models, RoPE must be applied before GQA expansion.

### B.5.3 Conv1D Orientation

GPT-2's Conv1D stores weights as (in, out). Must transpose.

### B.5.4 Missing `is_nested_encoder` in Output

If nested but you apply `out_proj`, you get double projection.

### B.5.5 LoRA on Frozen SVD

If LoRA isn't initialized to zero, the initial output differs from the 
pretrained model.

---