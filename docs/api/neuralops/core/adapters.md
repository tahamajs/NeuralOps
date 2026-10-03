Module neuralops.core.adapters
==============================
Model‑family adapters. Extracts projection weights and biases cleanly across all Transformer architectures.

Functions
---------

`get_adapter(arch: str) ‑> Any`
:   Return universal adapter for any architecture.

Classes
-------

`UniversalAdapter()`
:   Universal adapter supporting BERT, DistilBERT, RoBERTa, LLaMA, GPT2, DeBERTa, etc.

    ### Static methods

    `get_kv_heads(config: Any) ‑> int`
    :   Return number of key-value heads for GQA/MQA.

    `get_rotary_emb(old_attn: Any) ‑> Any | None`
    :   Return rotary embeddings if present.

    `get_weight_bias(old_attn: Any, proj: str) ‑> tuple`
    :   Extract (weight, bias) for projection 'q', 'k', 'v', or 'o'.
        
        Handles:
        - RoBERTa/BERT: query/key/value attributes on .self sub-module
        - LLaMA/Qwen: q_proj/k_proj/v_proj on attention module directly
        - GPT-2: c_attn (fused, already handled by fused-QKV path); c_proj
        - DistilBERT: q_lin/k_lin/v_lin/out_lin
        - LoRALinear: unwraps .base_layer to get the frozen base weight
        - Conv1D: transposes weight from [in, out] to [out, in]

`DistilBertAdapter()`
:   Universal adapter supporting BERT, DistilBERT, RoBERTa, LLaMA, GPT2, DeBERTa, etc.

    ### Static methods

    `get_kv_heads(config: Any) ‑> int`
    :   Return number of key-value heads for GQA/MQA.

    `get_rotary_emb(old_attn: Any) ‑> Any | None`
    :   Return rotary embeddings if present.

    `get_weight_bias(old_attn: Any, proj: str) ‑> tuple`
    :   Extract (weight, bias) for projection 'q', 'k', 'v', or 'o'.
        
        Handles:
        - RoBERTa/BERT: query/key/value attributes on .self sub-module
        - LLaMA/Qwen: q_proj/k_proj/v_proj on attention module directly
        - GPT-2: c_attn (fused, already handled by fused-QKV path); c_proj
        - DistilBERT: q_lin/k_lin/v_lin/out_lin
        - LoRALinear: unwraps .base_layer to get the frozen base weight
        - Conv1D: transposes weight from [in, out] to [out, in]

`BertAdapter()`
:   Universal adapter supporting BERT, DistilBERT, RoBERTa, LLaMA, GPT2, DeBERTa, etc.

    ### Static methods

    `get_kv_heads(config: Any) ‑> int`
    :   Return number of key-value heads for GQA/MQA.

    `get_rotary_emb(old_attn: Any) ‑> Any | None`
    :   Return rotary embeddings if present.

    `get_weight_bias(old_attn: Any, proj: str) ‑> tuple`
    :   Extract (weight, bias) for projection 'q', 'k', 'v', or 'o'.
        
        Handles:
        - RoBERTa/BERT: query/key/value attributes on .self sub-module
        - LLaMA/Qwen: q_proj/k_proj/v_proj on attention module directly
        - GPT-2: c_attn (fused, already handled by fused-QKV path); c_proj
        - DistilBERT: q_lin/k_lin/v_lin/out_lin
        - LoRALinear: unwraps .base_layer to get the frozen base weight
        - Conv1D: transposes weight from [in, out] to [out, in]

`LlamaAdapter()`
:   Universal adapter supporting BERT, DistilBERT, RoBERTa, LLaMA, GPT2, DeBERTa, etc.

    ### Static methods

    `get_kv_heads(config: Any) ‑> int`
    :   Return number of key-value heads for GQA/MQA.

    `get_rotary_emb(old_attn: Any) ‑> Any | None`
    :   Return rotary embeddings if present.

    `get_weight_bias(old_attn: Any, proj: str) ‑> tuple`
    :   Extract (weight, bias) for projection 'q', 'k', 'v', or 'o'.
        
        Handles:
        - RoBERTa/BERT: query/key/value attributes on .self sub-module
        - LLaMA/Qwen: q_proj/k_proj/v_proj on attention module directly
        - GPT-2: c_attn (fused, already handled by fused-QKV path); c_proj
        - DistilBERT: q_lin/k_lin/v_lin/out_lin
        - LoRALinear: unwraps .base_layer to get the frozen base weight
        - Conv1D: transposes weight from [in, out] to [out, in]