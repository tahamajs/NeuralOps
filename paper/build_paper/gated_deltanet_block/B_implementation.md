<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `try/except` Pattern

```python
try:
    adapter = UniversalAdapter()
    w_q, b_q = adapter.get_weight_bias(old_attn, "q")
    ...
    qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
    if qkv_w.shape == self.attn.in_proj_weight.shape:
        self.attn.in_proj_weight.copy_(qkv_w)
    ...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass
```

**Problems:**
1. Catches 5 exception types
2. Silently continues on failure
3. No logging
4. No verification

**Fix:**

```python
try:
    ...
except Exception as e:
    logger.warning(f"Weight transfer failed: {e}")
    raise  # ← do not swallow
```

## B.2 The Nested Encoder Path

```python
if self.is_nested_encoder:
    return delta_out, None  # ← no LN, no residual
```

For RoBERTa, this returns raw context. The outer `BertSelfOutput` applies 
residual + LN.

**Verification:**

```python
# Check that self.norm is not used
def forward(self, x):
    if self.is_nested_encoder:
        assert not hasattr(self, '_norm_used')  # should not be touched
        return delta_out, None
```

## B.3 Causal Masking for GPT-2

The `use_full_attention=True` path does not pass `attn_mask`:

```python
attn_out, attn_weights = self.attn(
    x, x, x, key_padding_mask=key_padding_mask, need_weights=output_attentions
)
```

**Fix for GPT-2:**

```python
if self.is_decoder:
    seq_len = x.shape[1]
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=x.device),
        diagonal=1
    )
    attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
```

## B.4 Shape Padding Logic

```python
if attn_out.shape[-1] != x.shape[-1]:
    attn_out = F.pad(attn_out, (0, x.shape[-1] - attn_out.shape[-1])) \
               if attn_out.shape[-1] < x.shape[-1] \
               else attn_out[..., :x.shape[-1]]
```

**Concern.** If shapes mismatch, this silently pads/truncates. **Should** 
instead be an assertion error — shape mismatch indicates a bug.

## B.5 Weight Transfer Logging

Add explicit logging:

```python
import logging
logger = logging.getLogger(__name__)

# After weight transfer:
qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
if qkv_w.shape == self.attn.in_proj_weight.shape:
    self.attn.in_proj_weight.copy_(qkv_w)
    logger.info(f"Q/K/V transfer OK: {qkv_w.shape}")
else:
    logger.error(f"Q/K/V shape mismatch: {qkv_w.shape} vs {self.attn.in_proj_weight.shape}")
```

## B.6 Common Pitfalls

### B.6.1 `UniversalAdapter` Returns None

If `get_weight_bias` returns `None`, `torch.cat([None, None, None])` raises. 
But `try/except` masks it.

### B.6.2 Nested Encoder Detection

`is_nested_encoder` must be set correctly. For DistilBERT, `False`. For 
RoBERTa, `True`. Getting this wrong leads to double LayerNorm or missing LN.

### B.6.3 LayerNorm Initialization

`nn.LayerNorm` defaults to $\gamma = 1, \beta = 0$. This is **not** 
pretrained. Consider copying pretrained LN if available.

## B.7 Registry

```python
@ATTENTION_REGISTRY.register("gated_deltanet_block")
class GatedDeltaNetBlock(BaseAttention):
    ...
```

## B.8 Injection Factory

Standard pattern (see `05_architecture.md`).

## B.9 Diagnostics

```python
def diagnose_block(model):
    for name, mod in model.named_modules():
        if isinstance(mod, GatedDeltaNetBlock):
            print(f"{name}:")
            print(f"  use_full_attention: {mod.use_full_attention}")
            print(f"  is_nested_encoder: {mod.is_nested_encoder}")
            print(f"  deltanet.is_nested_encoder: {mod.deltanet.is_nested_encoder}")
            if mod.attn is not None:
                w = mod.attn.in_proj_weight
                print(f"  in_proj_weight norm: {w.norm().item():.4f}")
```

---