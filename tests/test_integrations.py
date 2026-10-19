# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for ecosystem and integration modules."""
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from neuralops.integrations.hf_automodel import (
    AutoNeuralOpsModel,
    AutoNeuralOpsForCausalLM,
    AutoNeuralOpsForSequenceClassification,
    register_hf_automodel,
    patch_model_attention,
)
from neuralops.integrations.serving import (
    VLLMAttentionAdapter,
    TGIAttentionAdapter,
    create_openai_serving_app,
)
from neuralops.integrations.distributed_compile import (
    get_deepspeed_config,
    apply_fsdp_wrapping,
    compile_attention_module,
)
from neuralops.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuralops.distributed.context_parallel import (
    ring_attention_chunk_exchange,
    RingLinearAttention,
)
from neuralops.inference.paged_cache import PagedKVCache, PagedStateTable
from neuralops.inference.speculative_decoding import (
    SelfSpeculativeDecoder,
    gamma_sweep_harness,
)
from neuralops.export.surgery_manifest import (
    SurgeryManifest,
    AttentionSurgeryEntry,
    CompressionSurgeryEntry,
)
from neuralops.export.serving import (
    export_serving_bundle,
    validate_serving_metadata,
)
from neuralops.export.exporter import export_model


def test_hf_automodel():
    assert isinstance(register_hf_automodel(), bool)
    mod1 = AutoNeuralOpsModel.from_pretrained("dummy_model", attention_type="full_attention")
    assert isinstance(mod1, nn.Module)

    mod2 = AutoNeuralOpsForCausalLM.from_pretrained("dummy_model", attention="full_attention")
    assert isinstance(mod2, nn.Module)

    mod3 = AutoNeuralOpsForSequenceClassification.from_pretrained("dummy_model")
    assert isinstance(mod3, nn.Module)


def test_serving_adapters():
    vllm = VLLMAttentionAdapter(attention_type="linear", head_dim=64, num_heads=4)
    meta = vllm.format_metadata(block_tables=[[1, 2]], context_lens=[10])
    assert meta["max_context_len"] == 10

    tgi = TGIAttentionAdapter(max_batch_size=2)
    assert len(tgi.prepare_batch([1, 2, 3, 4])) == 2

    app = create_openai_serving_app()
    assert app is not None
    assert app.title == "NeuralOps Serving Engine"


def test_distributed_and_compile():
    ds_cfg = get_deepspeed_config(stage=3)
    assert ds_cfg["zero_optimization"]["stage"] == 3

    lin = nn.Linear(8, 8)
    fsdp_mod = apply_fsdp_wrapping(lin)
    assert isinstance(fsdp_mod, nn.Module)

    comp_mod = compile_attention_module(lin)
    assert isinstance(comp_mod, nn.Module)


def test_tensor_parallel_linear():
    col = ColumnParallelLinear(in_features=16, out_features=32, tp_size=1)
    x = torch.randn(2, 4, 16)
    out_col = col(x)
    assert out_col.shape == (2, 4, 32)

    row = RowParallelLinear(in_features=32, out_features=16, tp_size=1)
    out_row = row(out_col)
    assert out_row.shape == (2, 4, 16)


def test_context_parallel_ring():
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    next_k, next_v = ring_attention_chunk_exchange(k, v, rank=0, world_size=1)
    assert torch.equal(k, next_k)
    assert torch.equal(v, next_v)

    ring_attn = RingLinearAttention(embed_dim=32, num_heads=2, ring_size=1)
    x = torch.randn(2, 8, 32)
    out = ring_attn(x)
    assert out.shape == (2, 8, 32)


def test_paged_cache_fork_and_cow():
    cache = PagedKVCache(num_blocks=16, block_size=4, num_heads=2, head_dim=8)
    cache.allocate_sequence(1)

    k1 = torch.randn(2, 8)
    v1 = torch.randn(2, 8)
    cache.append_token(1, k1, v1)
    assert cache.seq_lengths[1] == 1

    forked_blocks = cache.fork_sequence(parent_seq_id=1, child_seq_id=2)
    assert cache.seq_lengths[2] == 1
    assert cache.block_ref_counts[forked_blocks[0]] == 2

    k2 = torch.randn(2, 8)
    v2 = torch.randn(2, 8)
    cache.append_token(2, k2, v2)
    assert cache.seq_lengths[2] == 2
    assert cache.block_tables[1][0] != cache.block_tables[2][0]

    cache.free_sequence(1)
    cache.free_sequence(2)
    assert len(cache.block_tables) == 0


def test_speculative_gamma_sweep():
    model = nn.Sequential(nn.Embedding(16, 16), nn.Linear(16, 16))
    sweep_results = gamma_sweep_harness(
        target_model=model,
        draft_model=model,
        input_ids=torch.randint(0, 16, (1, 4)),
        gamma_values=[1, 2],
        max_new_tokens=4,
    )
    assert 1 in sweep_results
    assert 2 in sweep_results
    assert sweep_results[1].total_tokens_generated >= 1


def test_manifest_hashing_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = SurgeryManifest(
            version="1.4.1",
            base_model_name="test_model",
            attentions=[AttentionSurgeryEntry(layer_path="layers.0", attention_type="cosformer")],
            compressions=[CompressionSurgeryEntry(layer_path="fc1", compressor_name="direct_svd", in_features=64, out_features=64, rank=8)],
        )
        assert manifest.validate_schema() is True
        sha = manifest.compute_sha256()
        assert len(sha) == 64

        ckpt_path = Path(tmpdir) / "manifest_ckpt.json"
        manifest.checkpoint(ckpt_path)
        assert ckpt_path.exists()

        loaded = SurgeryManifest.load_checkpoint(ckpt_path)
        assert loaded.compute_sha256() == sha


def test_export_serving_bundle_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = nn.Sequential(nn.Linear(16, 16))
        bundle_path = export_serving_bundle(model, output_dir=tmpdir, target_backend="vllm")
        is_valid, msg = validate_serving_metadata(bundle_path)
        assert is_valid is True, msg

        awq_dir = Path(tmpdir) / "awq_bundle"
        export_model(model, output_dir=awq_dir, format="awq")
        assert (awq_dir / "quant_config.json").exists()

        gguf_dir = Path(tmpdir) / "gguf_bundle"
        export_model(model, output_dir=gguf_dir, format="gguf")
        assert (gguf_dir / "gguf_metadata.json").exists()
