# 5. Architecture and Implementation
The module instantiates:
- `q_proj`, `k_proj`, `v_proj`, `out_proj`: Wrapped via `LoRALinear` with rank $r=16$ <!-- source: Codes/framework/attentions/adaptive_span_attention.py lines 85, 273-277 -->
- `span_logits`: `nn.Parameter` of shape $(H,)$ <!-- source: Codes/framework/attentions/adaptive_span_attention.py line 282 -->
- Support for Rotary Position Embeddings (`apply_rotary_pos_emb`) and GQA head repeating (`repeat_kv`)
