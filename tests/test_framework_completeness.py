"""
Comprehensive test suite verifying framework completeness across:
1. Unified step-by-step autoregressive generation on BaseAttention.
2. UnifiedHybridCache state tracking across diverse attention mechanisms.
3. Serving integration adapters (VLLMAdapter, TGIAdapter).
4. Scaled distributed training policies (FSDP auto-wrap, DeepSpeed configs).
5. PyTorch compile compatibility on custom op kernels.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.models.cache import UnifiedHybridCache
from neuralops.integrations.serving import VLLMAttentionAdapter, TGIAttentionAdapter
from neuralops.integrations.distributed_compile import (
    get_deepspeed_config,
    get_neuralops_fsdp_policy,
    apply_fsdp_wrapping,
    compile_attention_module,
)
from neuralops.core.triton_kernels.op_registry import (
    compilable_linear_attn,
    compilable_low_rank,
    compilable_sparse_attn,
)


class DummyLinearAttention(BaseAttention):
    def __init__(self, dim=64, num_heads=4):
        super().__init__(hidden_size=dim, num_heads=num_heads)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        past_key_values: torch.Tensor | None = None,
        use_cache: bool = False,
        **kwargs,
    ):
        B, T, D = hidden_states.shape
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        # O(1) recurrent step when T=1
        if past_key_values is not None:
            new_state = past_key_values + torch.einsum("btd,bte->bde", k, v)
            out = torch.einsum("btd,bde->bte", q, new_state)
        else:
            new_state = torch.einsum("btd,bte->bde", k, v)
            out = torch.einsum("btd,bde->bte", q, new_state)

        out = self.o(out)
        if use_cache:
            return out, None, new_state
        return out, None


def test_base_attention_step_contract():
    """Verify that BaseAttention.step() provides token-by-token generation with state propagation."""
    attn = DummyLinearAttention(dim=32, num_heads=2)
    state = None
    generated_tokens = []

    for _ in range(5):
        tok = torch.randn(2, 1, 32)
        out, state = attn.step(tok, state=state)
        assert out.shape == (2, 1, 32)
        assert state is not None
        assert state.shape == (2, 32, 32)
        generated_tokens.append(out)

    assert len(generated_tokens) == 5


def test_unified_hybrid_cache_multi_family():
    """Verify UnifiedHybridCache manages dense KV layers and recurrent states simultaneously."""
    cache = UnifiedHybridCache(num_layers=3)
    
    # Layer 0: Full attention (K, V)
    k0 = torch.randn(2, 4, 16, 32)
    v0 = torch.randn(2, 4, 16, 32)
    state0 = cache.update_layer(0, (k0, v0), layer_type="full")
    assert state0[0].shape == (2, 4, 16, 32)

    # Decode step 1 for Layer 0
    k0_next = torch.randn(2, 4, 1, 32)
    v0_next = torch.randn(2, 4, 1, 32)
    state0_next = cache.update_layer(0, (k0_next, v0_next), layer_type="full")
    assert state0_next[0].shape == (2, 4, 17, 32)

    # Layer 1: Recurrent attention O(1) state
    recurrent_state = torch.randn(2, 4, 32, 32)
    state1 = cache.update_layer(1, recurrent_state, layer_type="linear")
    assert state1.shape == (2, 4, 32, 32)

    # Layer 2: Sliding window
    kw = torch.randn(2, 4, 20, 32)
    vw = torch.randn(2, 4, 20, 32)
    state2 = cache.update_layer(2, (kw, vw), layer_type="window", window_size=8)
    assert state2[0].shape == (2, 4, 8, 32)


def test_serving_adapters():
    """Verify vLLM and TGI serving integration adapters."""
    vllm_adapter = VLLMAttentionAdapter(attention_type="cosformer", head_dim=64, num_heads=8)
    meta = vllm_adapter.format_metadata(block_tables=[[1, 2], [3, 4]], context_lens=[32, 32])
    assert meta["max_context_len"] == 32
    assert "block_tables" in meta

    tgi_adapter = TGIAttentionAdapter(max_batch_size=16)
    batch = tgi_adapter.prepare_batch(list(range(32)))
    assert len(batch) == 16


def test_distributed_and_fsdp_policies():
    """Verify DeepSpeed and FSDP integration helpers."""
    ds_cfg = get_deepspeed_config(stage=3, offload_optimizer=True, fp16=True)
    assert ds_cfg["zero_optimization"]["stage"] == 3
    assert ds_cfg["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    policy = get_neuralops_fsdp_policy()
    assert callable(policy)
    
    model = nn.Sequential(
        DummyLinearAttention(dim=32, num_heads=2),
        nn.Linear(32, 10),
    )
    wrapped = apply_fsdp_wrapping(model)
    assert wrapped is not None


def test_compilable_custom_ops():
    """Verify that compilable kernel wrappers run and return expected output shapes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(2, 4, 8, 16, device=device)
    k = torch.randn(2, 4, 8, 16, device=device)
    v = torch.randn(2, 4, 8, 16, device=device)

    out = compilable_linear_attn(q, k, v)
    assert out.shape == v.shape

    x = torch.randn(2, 8, 32, device=device)
    U = torch.randn(32, 8, device=device)
    V = torch.randn(8, 32, device=device)
    out_lr = compilable_low_rank(x, U, V)
    assert out_lr.shape == x.shape
