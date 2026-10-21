#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for ecosystem and integration modules."""
import tempfile
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from neuralops.integrations.hf_automodel import _BaseAutoNeuralOps
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
    VLLMAdapter as ExportVLLMAdapter,
    export_serving_bundle,
    validate_serving_metadata,
)
from neuralops.export.exporter import export_model
from neuralops.integrations.compressed_tensors import CompressedTensorsExporter


def test_hf_loader_propagates_checkpoint_errors_instead_of_substituting_random_model():
    class BrokenLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError("checkpoint unavailable")

    class TestAutoModel(_BaseAutoNeuralOps):
        _HF_CLASS = BrokenLoader

    with pytest.raises(OSError, match="checkpoint unavailable"):
        TestAutoModel.from_pretrained("missing-checkpoint")


def test_serving_adapters():
    vllm = VLLMAttentionAdapter(attention_type="linear", head_dim=64, num_heads=4)
    meta = vllm.format_metadata(block_tables=[[1, 2]], context_lens=[10])
    assert meta["max_context_len"] == 10

    tgi = TGIAttentionAdapter(max_batch_size=2)
    assert len(tgi.prepare_batch([1, 2, 3, 4])) == 2

    app = create_openai_serving_app()
    assert app is not None
    assert app.title == "NeuralOps Serving Engine"


def test_distributed_and_compile(monkeypatch):
    ds_cfg = get_deepspeed_config(stage=3)
    assert ds_cfg["zero_optimization"]["stage"] == 3
    with pytest.raises(ValueError, match="stage"):
        get_deepspeed_config(stage=4)
    with pytest.raises(ValueError, match="optimizer offload"):
        get_deepspeed_config(stage=0, offload_optimizer=True)

    lin = nn.Linear(8, 8)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized"):
        apply_fsdp_wrapping(lin)

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


def test_context_parallel_ring(monkeypatch):
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    next_k, next_v = ring_attention_chunk_exchange(k, v, rank=0, world_size=1)
    assert torch.equal(k, next_k)
    assert torch.equal(v, next_v)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized"):
        ring_attention_chunk_exchange(k, v, rank=0, world_size=2)

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
            version="1.4.3",
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
        assert "unverified" in msg

        awq_dir = Path(tmpdir) / "awq_bundle"
        with pytest.raises(NotImplementedError, match="does not quantize a model"):
            export_model(model, output_dir=awq_dir, format="awq")
        assert not awq_dir.exists()

        gguf_dir = Path(tmpdir) / "gguf_bundle"
        with pytest.raises(NotImplementedError, match="GGUF export is not implemented"):
            export_model(model, output_dir=gguf_dir, format="gguf")
        assert not gguf_dir.exists()


def test_export_serving_bundle_rejects_unknown_backend_before_writing(tmp_path):
    model = nn.Linear(4, 4)
    target = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="Unsupported serving backend"):
        export_serving_bundle(model, output_dir=target, target_backend="made_up")
    assert not target.exists()


def test_export_serving_bundle_does_not_claim_to_quantize(tmp_path):
    target = tmp_path / "unquantized"
    with pytest.raises(NotImplementedError, match="does not quantize"):
        export_serving_bundle(nn.Linear(4, 4), output_dir=target, quantization="int4")
    assert not target.exists()


def test_export_model_rejects_unknown_format_before_writing(tmp_path):
    target = tmp_path / "unsupported-format"
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_model(nn.Linear(4, 4), output_dir=target, format="made_up")
    assert not target.exists()


def test_compressed_tensors_exporter_rejects_dense_weights_without_quant_metadata(tmp_path):
    target = tmp_path / "false-quantized-export"
    with pytest.raises(ValueError, match="will not label ordinary dense weights"):
        CompressedTensorsExporter().export(nn.Linear(4, 4), str(target))
    assert not target.exists()


def test_paged_attention_rejects_unsupported_query_layout_before_backend_import():
    with pytest.raises(ValueError, match="query must have shape"):
        ExportVLLMAdapter.paged_attention_forward(
            torch.zeros(2, 3, 4, 5, 6),
            torch.zeros(2, 16, 4, 6),
            torch.zeros(2, 16, 4, 6),
            torch.zeros(1, 2, dtype=torch.int32),
            torch.tensor([16]),
        )


def test_generic_paged_cache_wrapper_fails_closed():
    with pytest.raises(NotImplementedError, match="cannot infer projected Q/K/V"):
        ExportVLLMAdapter.wrap_forward_for_paged_kv(nn.Identity())


def test_vllm_backend_registration_requires_the_real_registry(monkeypatch):
    import neuralops.export.serving as serving_module

    class Backend:
        @staticmethod
        def get_name():
            return "NEURALOPS_TEST"

    monkeypatch.setattr(serving_module, "HAS_VLLM", True)
    monkeypatch.setitem(sys.modules, "vllm.attention.backends.registry", None)
    with pytest.raises(RuntimeError, match="registry API"):
        serving_module.VLLMAdapter.register_attention_backend("NEURALOPS_TEST", Backend)


def test_onnx_export_failure_does_not_leave_a_fake_artifact(tmp_path: Path, monkeypatch):
    model = nn.Sequential(nn.Linear(4, 4)).eval()

    def fail_export(*args, **kwargs):
        raise ValueError("test exporter failure")

    monkeypatch.setattr(torch.onnx, "export", fail_export)

    with pytest.raises(RuntimeError, match="cosformer") as exc_info:
        export_model(
            model,
            output_dir=tmp_path / "onnx_failure",
            format="onnx",
            attention_type="cosformer",
            sample_input=torch.randn(1, 4),
        )

    failed_dir = tmp_path / "onnx_failure"
    assert "test exporter failure" in str(exc_info.value)
    assert not (failed_dir / "model.onnx").exists()
    assert not list(failed_dir.glob(".neuralops-onnx-*.onnx"))
