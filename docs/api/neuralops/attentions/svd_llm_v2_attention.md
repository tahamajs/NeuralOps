Module neuralops.attentions.svd_llm_v2_attention
================================================
SVD-LLM V2 Attention: Truncation-Aware SVD for LLM Compression.
Direct low-rank factorized parameterization maintaining explicit U, Sigma, and V^H factor buffers,
with trainable LoRA adapters for recovery fine-tuning.

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None, unsqueeze_dim: int = 1) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) with partial dimension support.

`inject_svd_llm_v2(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, rank: int | None = 192, **kwargs: Any) ‑> Any | None`
:   Safely constructs and injects SVDLLMV2Attention into a model or transformer layer.

Classes
-------

`SVDLLMV2Attention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = 192, energy_threshold: float = 0.999, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, dropout: float = 0.1, **kwargs: Any)`
:   SVD-factorized attention layer with explicit U, Sigma, V^H matrix buffers
    and trainable LoRA recovery adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.