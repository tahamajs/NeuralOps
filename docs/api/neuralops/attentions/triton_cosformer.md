Module neuralops.attentions.triton_cosformer
============================================
Triton-Accelerated Cosformer Linear Attention Kernel.

Implements the Cosformer algorithm (Zheng et al., 2022) with hardware-fused OpenAI Triton kernels:
  1. In-Register Cosine & Sine Positional Reweighting:
     - Directly computes cos(pi * i / 2M) and sin(pi * i / 2M) inside GPU registers (tl.cos, tl.sin).
     - Uses fixed max_length (M = max_position_embeddings) from model config for consistent invariant geometry.
     - Completely bypasses intermediate DRAM weight buffer allocations.

  2. Fused Chunked Causal Cosformer Attention (_triton_chunked_cosformer_fwd):
     - Intra-chunk attention score calculation in SRAM tiles.
     - Dual inter-chunk state tracking (S_cos, S_sin, Z_cos, Z_sin) in FP32 registers.
     - True O(N * d^2) linear complexity with causal lower-triangular masking.

  3. Bidirectional Cosformer Kernel (_triton_bidirectional_cosformer_fwd):
     - Global parallel reduction over sequence length for encoder backbones (BERT, RoBERTa).

  4. Full Autograd Integration (_TritonCosformerAutogradFunction):
     - Uses explicit torch.autograd.grad(out, [q_req, k_req, v_req], grad_out) without double-backward warnings.
     - Ensures complete differentiability for end-to-end training and recovery fine-tuning.

  5. Pure O(N) Fallback for CPU / Mac MPS:
     - Evaluates causal attention using chunked streaming prefix sums (chunk_size=64).
     - Strictly eliminates O(N^2) memory footprint on non-CUDA platforms.

  6. Production-Ready Weight Loading & RoPE:
     - Full copy_weights supporting GPT-2 (fused Conv1D c_attn), RoBERTa/BERT, and flat linear layers.
     - Complete Rotary Position Embeddings (RoPE) and Grouped Query Attention (GQA) support.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_triton_cosformer(model: nn.Module = None, rank: int = 64, copy_weights: bool = True, **kwargs: Any) ‑> torch.nn.modules.module.Module | None`
:   Safe factory function to inject TritonCosformerAttention into a model.

`triton_cosformer_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_decoder: bool = False, eps: float = 1e-06, max_length: int = 2048) ‑> torch.Tensor`
:   Convenience autograd functional wrapper for Triton Cosformer.

Classes
-------

`TritonCosformerAttention(config: Any = None, embed_dim: int | None = None, num_heads: int | None = None, num_kv_heads: int | None = None, is_decoder: bool = False, eps: float = 1e-06, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiability,
    RoPE support, Pretrained Weight Loading, and Masked Attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `triton_kernel`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Transfer pretrained projection weights from GPT-2, RoBERTa/BERT, or standard linear layers.

    `copy_weights_from_pretrained(self, src: Any) ‑> None`
    :   Transfer pretrained projection weights from GPT-2, RoBERTa/BERT, or standard linear layers.