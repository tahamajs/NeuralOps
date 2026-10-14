import pytest
import torch
import torch.nn as nn

from neuralops.integrations.vllm import (
    NeuralOpsvLLMMetadata,
    NeuralOpsvLLMImpl,
    NeuralOpsvLLMAttentionBackend,
    NeuralOpsForCausalLMvLLM,
    register_neuralops_vllm_plugin,
)
from neuralops.integrations.serving import (
    VLLMAttentionAdapter,
    TGIAttentionAdapter,
    SGLangRadixLinearCacheAdapter,
)


def test_vllm_backend_and_metadata_interface():
    assert NeuralOpsvLLMAttentionBackend.get_name() == "NEURALOPS_HYBRID"
    assert NeuralOpsvLLMAttentionBackend.get_impl_cls() is NeuralOpsvLLMImpl
    assert NeuralOpsvLLMAttentionBackend.get_metadata_cls() is NeuralOpsvLLMMetadata

    shape = NeuralOpsvLLMAttentionBackend.get_kv_cache_shape(
        num_blocks=32, block_size=16, num_kv_heads=4, head_size=64
    )
    assert shape == (32, 4, 64, 64)


def test_vllm_impl_prefill_and_decode_step():
    impl = NeuralOpsvLLMImpl(num_heads=2, head_size=16, scale=1.0 / 4.0)

    # 1. Prefill pass
    B, T, H, D = 2, 8, 2, 16
    q = torch.randn(B, T, H * D)
    k = torch.randn(B, T, H * D)
    v = torch.randn(B, T, H * D)

    meta_prefill = NeuralOpsvLLMMetadata(
        num_prefill_tokens=B * T,
        num_decode_tokens=0,
        seq_ids=[0, 1],
        page_table=torch.zeros((B, 4), dtype=torch.int32),
        slot_mapping=torch.zeros((B, T), dtype=torch.int32),
        context_lens=torch.tensor([T, T], dtype=torch.int32),
        is_prompt=True,
    )
    out_prefill = impl.forward(q, k, v, kv_cache=None, attn_metadata=meta_prefill)
    assert out_prefill.shape == (B, T, H * D)

    # 2. Continuous batching decode pass with paged state pool
    total_pages = 8
    paged_state_pool = torch.zeros((total_pages, H, D, D))
    block_tables = torch.tensor([[1], [3]], dtype=torch.int32)

    q_dec = torch.randn(B, 1, H * D)
    k_dec = torch.randn(B, 1, H * D)
    v_dec = torch.randn(B, 1, H * D)

    meta_decode = NeuralOpsvLLMMetadata(
        num_prefill_tokens=0,
        num_decode_tokens=B,
        seq_ids=[0, 1],
        page_table=block_tables,
        slot_mapping=torch.zeros((B, 1), dtype=torch.int32),
        context_lens=torch.tensor([T + 1, T + 1], dtype=torch.int32),
        is_prompt=False,
    )

    out_dec = impl.forward(q_dec, k_dec, v_dec, kv_cache=paged_state_pool, attn_metadata=meta_decode)
    assert out_dec.shape == (B, 1, H * D)
    # Ensure physical pages in state pool were updated non-trivially
    assert paged_state_pool[1].abs().sum() > 0
    assert paged_state_pool[3].abs().sum() > 0


def test_vllm_model_wrapper_and_plugin_registration():
    class DummyConfig:
        hidden_size = 32
        d_model = 32
        vocab_size = 100
        num_attention_heads = 2
        num_hidden_layers = 1

    model = NeuralOpsForCausalLMvLLM(DummyConfig())
    input_ids = torch.randint(0, 100, (2, 8))
    hidden = model(input_ids)
    assert hidden.shape == (2, 8, 32)
    logits = model.compute_logits(hidden)
    assert logits.shape == (2, 8, 100)

    # Plugin registration hook (safe fallback when vllm not installed)
    res = register_neuralops_vllm_plugin()
    assert isinstance(res, bool)


def test_sglang_radix_linear_cache_adapter():
    cache = SGLangRadixLinearCacheAdapter(max_cached_nodes=4)

    # Cache prompt A
    tokens_a = [101, 2054, 2003, 1037]
    state_a = torch.randn(2, 4, 16, 16)
    cache.cache_finished_req(tokens_a, state_a)

    # Match exact prefix
    matched_len, matched_state = cache.match_prefix(tokens_a)
    assert matched_len == 4
    assert torch.equal(matched_state, state_a)

    # Match sub-prefix
    query_tokens = [101, 2054, 2003, 1037, 5000, 6000]
    matched_len, matched_state = cache.match_prefix(query_tokens)
    assert matched_len == 4
    assert torch.equal(matched_state, state_a)

    # Lock reference and eviction test
    cache.inc_lock_ref(tokens_a)
    tokens_b = [1, 2, 3]
    cache.cache_finished_req(tokens_b, torch.randn(1))
    evicted = cache.evict(1)
    assert evicted == 1
    # tokens_a should still exist because ref_count > 0
    assert tuple(tokens_a) in cache.tree
    assert tuple(tokens_b) not in cache.tree

    cache.dec_lock_ref(tokens_a)
    cache.evict(1)
    assert tuple(tokens_a) not in cache.tree
