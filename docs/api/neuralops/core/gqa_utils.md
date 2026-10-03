Module neuralops.core.gqa_utils
===============================
Shared Grouped-Query Attention (GQA) & Multi-Head Attention (MHA) Utilities.
Provides dynamic dimension extraction, head repetition, and reshaping
for Qwen3.5, LLaMA, Mistral, Gemma, and legacy MHA models.

Functions
---------

`get_qkv_projection_dims(adapter: Any) ‑> tuple[int, int, int, int]`
:   Extracts (q_proj_out, kv_proj_out, head_dim, kv_repeat) dynamically from adapter or config.
    Supports dictionary adapters, HF PretrainedConfig, or object adapters.
    Returns:
        q_out: output dimension for query projection (e.g. 2048 for Qwen3.5-0.8B)
        kv_out: output dimension for key/value projection (e.g. 512 for Qwen3.5-0.8B)
        head_dim: dimension of each head (e.g. 256 for Qwen3.5-0.8B)
        n_rep: repetition factor kv_repeat = num_heads // num_key_value_heads (e.g. 4 for Qwen3.5-0.8B)

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Expands key/value heads from num_key_value_heads to num_attention_heads.
    Equivalent to torch.repeat_interleave(hidden_states, dim=1, repeats=n_rep)
    Shape: (batch, num_key_value_heads, seq_len, head_dim) -> (batch, num_attention_heads, seq_len, head_dim)
    For standard MHA where n_rep <= 1, returns hidden_states directly (zero-copy).

`reshape_for_attention(x: torch.Tensor, num_heads: int, head_dim: int) ‑> torch.Tensor`
:   Reshapes projected tensor (B, N, num_heads * head_dim) -> (B, num_heads, N, head_dim)
    Safe for both MHA and GQA query / key / value representations.