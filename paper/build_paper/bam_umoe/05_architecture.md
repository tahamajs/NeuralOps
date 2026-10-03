# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | Cross-attn? |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | No |
| RoBERTa | Nested | Yes | No | No | No |
| GPT-2 | Causal | No | No | No | No |
| Qwen3.5 | GQA+RoPE | No | Yes | Yes | No |
| T5 | Enc-Dec | No | No | No | **Yes** |

## 5.2 Weight Extraction

### DistilBERT
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### RoBERTa (Nested)
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight  # For reference, not applied
is_nested_encoder = True
```

### GPT-2 (Fused Conv1D)
```python
fused = src.c_attn.weight.T
q_w, k_w, v_w = fused.chunk(3, dim=0)
o_w = src.c_proj.weight.T
```

### Qwen (GQA + RoPE)
Direct copy from `q_proj`, `k_proj`, `v_proj`, `o_proj`.

## 5.3 Nested Encoder Bypass

For RoBERTa, BAM-UMoE's `out_lin` is not applied; the outer 
`BertSelfOutput.dense` performs the projection. This avoids double projection.

## 5.4 Partial RoPE

For Qwen3.5 with `head_dim=256` and `rotary_dim=64`:
```python
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
q = torch.cat([q_rot, q_pass], dim=-1)
```

## 5.5 Cross-Attention

For encoder-decoder models, the MoE branch is **not applied** to 
cross-attention. The reason: cross-attention V is projected from encoder 
states, and pretrained cross-attention weights are highly calibrated. 
Injecting unaligned MoE there would disrupt alignment.

```python
if not is_cross_attention:
    moe_delta_v = self.qkv_moe(hidden_states).view(B, L, H, d_h)
    v = v + 0.1 * moe_delta_v
```

## 5.6 Registry

```python
@ATTENTION_REGISTRY.register("bam_umoe")
class BAMUMoEBlock(BaseAttention): ...
```

## 5.7 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Return None |
| NaN in scores | `torch.nan_to_num` | Replace with 0 |
| MoE overflow | `torch.clamp` | Clamp to ±65000 |
| Cross-attention MoE | Skip MoE branch | Preserve alignment |

## 5.8 Numerical Stability

| Component | Precision |
|---|---|
| Attention scores | FP32 |
| MoE router | FP32 |
| MoE expert | Model dtype |
| Output | Model dtype |

---