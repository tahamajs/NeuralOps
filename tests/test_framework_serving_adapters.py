"""
Exhaustive tests for verified, existing serving and framework integrations:
1. PagedStateTable from neuralops.models.paged_state (vLLM-compatible in-tree state table)
2. Hugging Face replace_attention() bit-exact roundtrip serialization via safetensors
3. ONNX export from neuralops.export with graph validation

Non-existent frameworks (TGI server, SGLang distributed runtime, TensorRT-LLM)
are intentionally omitted per forensic code audit.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

import neuralops
from neuralops.models.paged_state import PagedStateTable
from neuralops.export.exporter import export_model, load_exported_model
from neuralops.integrations.hf_automodel import register_hf_automodel


def test_paged_state_table_lifecycle_and_cow():
    """Verify PagedStateTable allocation, zero-copy sequence forking, and Copy-on-Write."""
    table = PagedStateTable(num_pages=16, num_heads=2, head_dim=16, device="cpu")
    assert len(table.free_pages) == 16

    # 1. Register parent sequence
    parent_pages = table.register_sequence(seq_id=10, initial_pages=2)
    assert len(parent_pages) == 2
    assert len(table.free_pages) == 14
    p0 = parent_pages[0]
    assert table.ref_counts[p0] == 1

    # Populate dummy state on parent page
    table.page_pool[p0, 0, 0, 0] = 77.0

    # 2. Zero-copy fork sequence to child
    child_pages = table.fork_sequence(parent_seq_id=10, child_seq_id=20)
    assert child_pages == parent_pages
    assert table.ref_counts[p0] == 2
    assert len(table.free_pages) == 14  # No new physical pages consumed

    # 3. Copy-on-Write (CoW) on mutating child sequence
    new_child_p0 = table.prepare_write(seq_id=20, page_slot=0)
    assert new_child_p0 != p0
    assert table.ref_counts[p0] == 1
    assert table.ref_counts[new_child_p0] == 1
    assert len(table.free_pages) == 13  # Allocated 1 new page for child
    assert table.page_pool[new_child_p0, 0, 0, 0] == 77.0  # Data cloned properly

    # 4. Block table tensor extraction
    block_tensor = table.get_block_table_tensor([10, 20], max_pages=4)
    assert block_tensor.shape == (2, 4)
    assert block_tensor[0, 0].item() == p0
    assert block_tensor[1, 0].item() == new_child_p0

    # 5. Free sequences and ensure all physical pages return to pool
    table.free_sequence(seq_id=10)
    assert p0 not in table.ref_counts
    table.free_sequence(seq_id=20)
    assert new_child_p0 not in table.ref_counts
    assert len(table.free_pages) == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_paged_state_table_cuda_allocation():
    """Verify PagedStateTable operates cleanly on GPU device."""
    table = PagedStateTable(num_pages=32, num_heads=4, head_dim=32, device="cuda")
    pages = table.register_sequence(seq_id=1, initial_pages=1)
    assert table.page_pool.device.type == "cuda"
    table.page_pool[pages[0]].fill_(3.14)
    table.free_sequence(seq_id=1)
    assert len(table.free_pages) == 32


def test_hf_replace_attention_safetensors_roundtrip(tmp_path: Path):
    """Verify bit-exact output reproduction across replace_attention and safetensors export/load."""
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()

    # In-place attention surgery
    neuralops.replace_attention(model, attention="cosformer")
    model.eval()

    export_dir = tmp_path / "hf_cosformer_export"
    export_model(model, export_dir, format="safetensors")

    assert (export_dir / "model.safetensors").exists()
    assert (export_dir / "neuralops_config.json").exists()

    # Reconstruct from fresh base model
    base_model = GPT2LMHeadModel(config).eval()
    loaded_model, meta = load_exported_model(export_dir, base_model=base_model)
    loaded_model.eval()

    assert meta.get("attention_type") == "cosformer"

    # Evaluate bit-exact equality on identical inputs
    torch.manual_seed(1234)
    sample_tokens = torch.randint(0, 100, (2, 12))

    with torch.no_grad():
        out_orig = model(sample_tokens).logits
        out_loaded = loaded_model(sample_tokens).logits

    max_diff = (out_orig - out_loaded).abs().max().item()
    assert max_diff == 0.0, f"Expected bit-exact match (0.0), got max diff {max_diff}"


def test_hf_automodel_registration_hook():
    """Verify HuggingFace AutoConfig and AutoModel registration hook."""
    success = register_hf_automodel()
    assert success is True

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from neuralops.models.configuration import NeuralOpsConfig

    assert "neuralops" in CONFIG_MAPPING
    assert CONFIG_MAPPING["neuralops"] is NeuralOpsConfig


def test_onnx_export_and_graph_validation(tmp_path: Path):
    """Verify ONNX export generates a topologically valid, inspectable ONNX computational graph."""
    import onnx
    from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor

    class TinyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(64, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.act(self.fc1(x)))

    model = TinyModule().eval()
    # Compress with Direct SVD
    comp = DirectSVDCompressor(model, rank=8, min_rank=4)
    comp.compress()
    model.eval()

    export_dir = tmp_path / "onnx_export"
    dummy_input = torch.randn(1, 32)
    onnx_file = export_model(model, export_dir, format="onnx", sample_input=dummy_input)

    assert onnx_file.exists()
    assert onnx_file.suffix == ".onnx"

    # Verify ONNX model protobuf validity
    onnx_model = onnx.load(str(onnx_file))
    onnx.checker.check_model(onnx_model)
    assert len(onnx_model.graph.node) > 0
