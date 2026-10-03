# Appendix B. Implementation Details

## B.1 Core Module

```python
class HGRN2GatedDeltaNet(BaseAttention):
    def __init__(self, config, rank=None, ...):
        super().__init__(config, rank)
        
        # Resolve dims
        self.d_model = ...
        self.num_heads = ...
        self.is_decoder = ...
        self.lora_rank = ...
        
        # Branches
        self.hgrn = HGRN2Layer(...)
        self.deltanet = GatedDeltaNet(...)
        
        # Gate
        self.gate = nn.Linear(self.d_model, self.d_model)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(0.0)
        
        # Output
        self.out_proj = LoRALinear(...)
        
        if old_attn is not None:
            self.copy_weights(old_attn)
```

## B.2 The Gate Initialization

```python
with torch.no_grad():
    self.gate.weight.zero_()
    self.gate.bias.fill_(0.0)
```

**Why?** Initializing to zero gives $\gamma = 0.5$, which is a neutral 
starting point. Random initialization would cause one branch to dominate 
arbitrarily.

## B.3 Failure-Bias

```python
if hgrn_ok and not deltanet_ok:
    with torch.no_grad():
        self.gate.weight.zero_()
        self.gate.bias.fill_(4.0)  # sigmoid(4) ≈ 0.98
```

When one branch's weight copy fails, we bias the gate toward the surviving 
branch. This prevents the failed branch from injecting random noise.

## B.4 The `sub_kwargs` Pattern

```python
sub_kwargs = {
    k: v
    for k, v in kwargs.items()
    if k not in ("is_decoder", "is_causal", "lora_rank", "lora_alpha")
}
```

This is essential to avoid the "multiple values for keyword argument" error 
when the same `kwargs` dict is passed to nested constructors.

## B.5 Output Projection Search

```python
if w_o is None:
    for cand in ("out_proj", "out_lin", "o_proj", "output", "proj", "c_proj", "dense"):
        for target in (old_attn, old_attn.self, old_attn.attention, old_attn.output):
            if target and hasattr(target, cand):
                mod = getattr(target, cand)
                if hasattr(mod, "dense"):
                    mod = mod.dense
                cw, cb = _unpack_weight_bias(mod)
                if cw is not None:
                    w_o, b_o = cw, cb
                    break
```

This search handles:
- DistilBERT: `out_lin`
- RoBERTa: `output.dense`
- GPT-2: `c_proj`
- Qwen: `o_proj`

## B.6 Double-Injection Guard

```python
if isinstance(curr_attn, HGRN2GatedDeltaNet):
    return None
if "HGRN2GatedDeltaNet" in curr_attn.__class__.__name__:
    return None
if hasattr(curr_attn, "inner") and isinstance(curr_attn.inner, HGRN2GatedDeltaNet):
    return None
if hasattr(curr_attn, "_wrapped_attention") and isinstance(
    curr_attn._wrapped_attention, HGRN2GatedDeltaNet
):
    return None
```

Four checks cover:
- Direct instance.
- Class name match.
- Wrapped inside `inner`.
- Wrapped inside `_wrapped_attention`.

## B.7 Skip Hybrid Layers

```python
if "DeltaNet" in c_name or "LinearAttention" in c_name:
    return None
```

Qwen3.5 has some layers using `DeltaNet` or `LinearAttention`. We skip them 
to avoid double-replacing recurrent modules.

## B.8 Cross-Attention Warning

```python
if encoder_hidden_states is not None:
    logger.warning(
        "HGRN2GatedDeltaNet received encoder_hidden_states but cross-attention "
        "is not implemented. Returning raw hidden_states (identity)."
    )
    return hidden_states, None
```

## B.9 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, **kwargs):
    # 1. Branch forwards
    hgrn_out = self.hgrn(hidden_states, attention_mask=attention_mask, **kwargs)[0]
    delta_out = self.deltanet(hidden_states, attention_mask=attention_mask, **kwargs)[0]
    
    # 2. Shape align
    if hgrn_out.shape != delta_out.shape:
        # pad
        ...
    
    # 3. Gate + fuse
    gamma = torch.sigmoid(self.gate(hidden_states))
    fused = gamma * hgrn_out + (1 - gamma) * delta_out
    
    # 4. Output
    out = self.out_proj(fused)
    
    return out, None
```

## B.10 Common Pitfalls

### B.10.1 Forgetting `sub_kwargs`

Passing the raw `kwargs` to branch constructors raises `TypeError`.

### B.10.2 Random Gate Init

Initializing `gate.weight` with default Kaiming gives arbitrary bias.

### B.10.3 Missing Out Projection Failure

Raising on out_proj failure is essential — otherwise the module has a 
random output projection.

### B.10.4 Double Injection

Without the guard, repeated calls to `inject_hgrn2_gated_deltanet` wrap the 
module multiple times.

---