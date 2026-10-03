Module neuralops.attentions.triton_svd_linear_attention
=======================================================
Triton-Accelerated SVD Linear Attention & Chunked Causal Linear Attention.

High-performance GPU implementation using OpenAI Triton language (tl) with:
  1. Chunked Causal Linear Attention Kernel (_triton_chunked_causal_linear_attn_fwd):
     - Keeps the recurrent key-value state S \in R^{D_k x D_v} strictly in GPU SRAM/registers.
     - Tiled in blocks of BLOCK_CHUNK = 64 to eliminate O(N^2) memory footprint.
     - Evaluates intra-chunk causal attention and inter-chunk prefix sums in a single fused pass.
     - Strictly O(N * d^2) FLOPs and O(N * d) memory.

  2. Bidirectional Linear Attention Kernel (_triton_bidirectional_linear_attn_fwd):
     - Fused parallel reduction of K^T V and K^T 1 for encoder architectures (DistilBERT, RoBERTa).
     - Fused query projection with numerical stability epsilon clamping in registers.

  3. Non-Zero Identity SVD Initialization (BUG 5 FIX):
     - SVDLinearProjection initializes U with identity-like structure (eye for rank dimensions)
       and V with Kaiming uniform so initial output is non-zero before fine-tuning.

  4. Backward Kernel & Clean Autograd (BUG 1 FIX):
     - Uses explicit torch.autograd.grad(out, [q_req, k_req, v_req], grad_out) preventing double-backward warnings.
     - Bridges _triton_linear_attn_bwd_kernel on CUDA GPUs for bidirectional attention.
     - Provides streaming chunked backward propagation for causal decoder attention.

  5. Attention Masking & Padding Invariant (BUG 2 FIX):
     - Ensures padding tokens on K and V are strictly zeroed out before attention accumulation,
       preventing contaminated normalization denominator Z.

  6. RoPE and GQA Support (BUG 3 & BUG 4 FIX):
     - Seamless Rotary Position Embeddings (RoPE) via apply_rotary_pos_emb.
     - Full Grouped Query Attention (GQA) support with separate num_kv_heads and repeat_kv.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_triton_svd_linear_attention(model: nn.Module = None, rank: int = 64, copy_weights: bool = True, **kwargs: Any) ‑> torch.nn.modules.module.Module | None`
:   Safe factory function to inject TritonSVDLinearAttention into a model.

`triton_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False, activation_mode: str = 'elu', eps: float = 1e-06) ‑> torch.Tensor`
:   Convenience functional wrapper for Triton Linear Attention.

Classes
-------

`SVDLinearProjection(in_features: int, out_features: int, rank: int = 64, bias: bool = True)`
:   Genuine SVD Low-Rank Linear Layer: W \approx U @ V.
    
    Factorizes in_features -> out_features through bottleneck rank r:
      V \in R^{r x in_features}  (Down-projection)
      U \in R^{out_features x r} (Up-projection)
      y = (x @ V^T) @ U^T + bias
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `triton_kernel`
    :   The type of the None singleton.

    ### Static methods

    `from_pretrained_linear(linear: nn.Linear, rank: int = 64) ‑> neuralops.attentions.triton_svd_linear_attention.SVDLinearProjection`
    :   Initializes low-rank factors directly from pretrained weights via Truncated SVD:
        W = U_full @ diag(S) @ V_full^T
        U = U_full[:, :r] @ sqrt(S[:r])
        V = sqrt(S[:r]) @ V_full^T[:r, :]

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `reset_parameters(self) ‑> None`
    :

`TritonSVDLinearAttention(config: Any = None, embed_dim: int | None = None, num_heads: int | None = None, num_kv_heads: int | None = None, rank: int | None = 64, is_decoder: bool = False, activation_mode: str = 'elu', eps: float = 1e-06, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Triton-Accelerated Linear Attention with Genuine SVD Low-Rank Projections,
    RoPE, GQA, and Masked Attention.
    
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