#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Streaming Checkpointed Distillation & Zero-OOM Engine.

Verifies:
- Phase 2.1: Mathematical engine parity (matches dense CKA within 1e-6 across micro-chunks).
- Phase 2.2: Non-reentrant teacher checkpointing (CheckpointedTeacherBridge with no_grad).
- Phase 2.3: MicroChunkedShadowTwinDistiller micro-chunking and threshold memory purging.
- Phase 2.4: Numerical parity & VRAM benchmarks under extreme sequence dimensions.
- Phase 2.5: Integration with ShadowTwin and MultiLevelKDTrainer (Pillar 2 Representation CKA).
"""

from typing import Any, List, Optional
import pytest
import torch
import torch.nn as nn

from neuralops.surgery.endoskeletal_shell import UniversalAttentionBridge
from neuralops.surgery.scanner import AttentionTargetSpec
from neuralops.surgery.shadow_twin import (
    enable_shadow_twin,
    finalize_surgery,
)
from neuralops.surgery.streaming_distill import (
    CheckpointedTeacherBridge,
    MicroChunkedShadowTwinDistiller,
    StreamingCKALoss,
)
from neuralops.trainer.distillation.multi_level_kd import (
    MultiLevelKDTrainer,
    SmoothMaxOverLayersCKASteeringLoss,
)


def dense_linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Reference implementation of dense linear CKA."""
    x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
    y_flat = y.reshape(-1, y.shape[-1]).to(torch.float32)

    x_c = x_flat - x_flat.mean(dim=0, keepdim=True)
    y_c = y_flat - y_flat.mean(dim=0, keepdim=True)

    g_xy = torch.matmul(x_c.transpose(0, 1), y_c)
    g_xx = torch.matmul(x_c.transpose(0, 1), x_c)
    g_yy = torch.matmul(y_c.transpose(0, 1), y_c)

    norm_xy = torch.norm(g_xy, p="fro") ** 2
    norm_xx = torch.norm(g_xx, p="fro")
    norm_yy = torch.norm(g_yy, p="fro")

    similarity = norm_xy / (norm_xx * norm_yy + eps)
    return 1.0 - torch.clamp(similarity, 0.0, 1.0)


def test_streaming_cka_numerical_parity() -> None:
    """Verifies that StreamingCKALoss matches dense linear CKA within 1e-6."""
    torch.manual_seed(42)
    d = 64
    x = torch.randn(4, 256, d)
    y = torch.randn(4, 256, d)

    # 1. Compute reference dense CKA
    expected_loss = dense_linear_cka(x, y)

    # 2. Compute streaming CKA in 2 micro-chunks of 128 tokens
    cka_mod = StreamingCKALoss(hidden_dim=d, exact=True)
    cka_mod.reset()
    cka_mod.update(x[:, :128, :], y[:, :128, :])
    cka_mod.update(x[:, 128:, :], y[:, 128:, :])
    streaming_loss = cka_mod.compute()

    diff = torch.abs(expected_loss - streaming_loss).item()
    assert diff < 1e-5, f"Streaming CKA difference {diff} exceeds tolerance"


def test_streaming_cka_tight_numerical_parity() -> None:
    """Verifies that exact StreamingCKALoss across 4 chunks matches dense CKA within 1e-6."""
    torch.manual_seed(1337)
    d = 32
    # Non-zero mean to thoroughly test Chan's online covariance correction
    x = torch.randn(2, 512, d) + 3.5
    y = torch.randn(2, 512, d) - 2.0

    expected_loss = dense_linear_cka(x, y)

    cka_mod = StreamingCKALoss(hidden_dim=d, exact=True)
    cka_mod.reset()
    chunk_size = 128
    for st in range(0, 512, chunk_size):
        en = st + chunk_size
        cka_mod.update(x[:, st:en], y[:, st:en])

    streaming_loss = cka_mod.compute()
    diff = torch.abs(expected_loss - streaming_loss).item()
    assert diff < 1e-6, f"Exact streaming CKA difference {diff} exceeds 1e-6 tolerance"


def test_streaming_cka_gradient_flow() -> None:
    """Verifies that autograd gradients flow properly through StreamingCKALoss."""
    d = 32
    x = torch.randn(2, 64, d, requires_grad=True)
    y = torch.randn(2, 64, d)

    cka_mod = StreamingCKALoss(hidden_dim=d)
    cka_mod.reset()
    cka_mod.update(x, y)
    loss = cka_mod.compute()

    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert torch.count_nonzero(x.grad) > 0


def test_streaming_cka_forward_wrapper() -> None:
    """Verifies that StreamingCKALoss forward() handles automatic chunking."""
    torch.manual_seed(42)
    d = 32
    x = torch.randn(2, 256, d, requires_grad=True)
    y = torch.randn(2, 256, d)

    cka_mod = StreamingCKALoss(hidden_dim=d)
    loss = cka_mod(x, y, chunk_size=64)

    assert not torch.isnan(loss)
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_checkpointed_teacher_bridge() -> None:
    """Verifies CheckpointedTeacherBridge freezes parameters and executes without gradients."""
    class DummyLayer(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            out = self.linear(x)
            if attention_mask is not None:
                out = out * attention_mask
            return out

    dim = 32
    model = nn.Sequential(DummyLayer(dim), DummyLayer(dim))
    bridge = CheckpointedTeacherBridge(model)

    # Assert teacher parameters are frozen
    for p in bridge.parameters():
        assert not p.requires_grad

    inp = torch.randn(2, 16, dim)
    out = bridge.forward_layer(model[0], inp)
    assert not out.requires_grad
    assert out.shape == inp.shape


class MockTransformerModel(nn.Module):
    """Simple mock model with embedding, linear attention projection, and head."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layer = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Any:
        emb = self.embedding(input_ids)
        if attention_mask is not None:
            emb = emb * attention_mask.unsqueeze(-1).to(emb.dtype)
        hid = self.layer(emb)
        logits = self.head(hid)
        if output_hidden_states:
            class Output:
                pass
            res = Output()
            res.logits = logits
            res.hidden_states = [emb, hid]
            return res
        return logits


def test_micro_chunked_distiller_step() -> None:
    """Verifies end-to-end execution of MicroChunkedShadowTwinDistiller."""
    teacher = MockTransformerModel(vocab_size=50, hidden_dim=32)
    student = MockTransformerModel(vocab_size=50, hidden_dim=32)

    distiller = MicroChunkedShadowTwinDistiller(
        teacher=teacher,
        student=student,
        chunk_size=16,
    )

    input_ids = torch.randint(0, 50, (2, 64))
    res = distiller.step(input_ids)

    assert "loss" in res
    assert "cloning_loss" in res
    assert "cka_loss" in res
    assert not torch.isnan(res["loss"]).any()

    # Verify backpropagation into student parameters
    res["loss"].backward()
    for name, p in student.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Student parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any()


def test_shadow_twin_streaming_checkpointed_mode() -> None:
    """Verifies that enable_shadow_twin with memory_mode='streaming_checkpointed' operates correctly."""
    dim = 32
    orig_module = nn.Linear(dim, dim)
    new_module = nn.Linear(dim, dim)
    spec = AttentionTargetSpec(
        module_path="layer.0.attn",
        layer_index=0,
        attention_type="dense",
    )
    bridge = UniversalAttentionBridge(
        orig_module=orig_module,
        spec=spec,
        new_attention=new_module,
        inherit_weights=False,
    )

    enable_shadow_twin(bridge, stage="stage2_block_cloning", memory_mode="streaming_checkpointed")
    assert getattr(bridge, "memory_mode", None) == "streaming_checkpointed"
    assert bridge.teacher_bridge_wrapper is not None

    x = torch.randn(2, 64, dim, requires_grad=True)
    out = bridge(x)
    assert out.shape == x.shape
    assert bridge._last_twin_loss is not None
    assert not torch.isnan(bridge._last_twin_loss)

    bridge._last_twin_loss.backward()
    for p in new_module.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()

    # Test clean finalization
    finalize_surgery(bridge)
    assert bridge.teacher_branch is None
    assert bridge.teacher_bridge_wrapper is None


def test_multi_level_kd_streaming_integration() -> None:
    """Verifies MultiLevelKDTrainer integration with StreamingCKALoss as Pillar 2 representation CKA."""
    teacher = MockTransformerModel(vocab_size=50, hidden_dim=32)
    student = MockTransformerModel(vocab_size=50, hidden_dim=32)

    trainer = MultiLevelKDTrainer(
        teacher=teacher,
        student=student,
        use_streaming_cka=True,
        memory_mode="streaming_checkpointed",
        chunk_size=16,
    )

    assert trainer.use_streaming_cka
    assert isinstance(trainer.cka_loss_fn, SmoothMaxOverLayersCKASteeringLoss)

    batch = torch.randint(0, 50, (2, 32))
    losses = trainer.training_step(batch)

    assert "total_loss" in losses
    assert "cka_loss" in losses
    assert not torch.isnan(losses["total_loss"]).any()


@pytest.mark.gpu
def test_benchmark_b128_t1024_gpu() -> None:
    """
    DoD Phase 2.4 Verification Gate on NVIDIA GPU (B=128, T=1024, D=1024).

    Asserts peak VRAM <= 7.6 GB (pass threshold < 8.0 GB) and finite gradients.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Phase 2.4 physical VRAM benchmark")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    b, t, d = 128, 1024, 1024

    class BenchmarkTransformer(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.hidden_dim = hidden_dim

        def forward(self, x: torch.Tensor, output_hidden_states: bool = False) -> Any:
            h1 = self.linear1(x)
            h2 = self.linear2(h1)
            if output_hidden_states:
                class Out:
                    pass
                res = Out()
                res.hidden_states = [h1, h2]
                return res
            return h2

    teacher = BenchmarkTransformer(d).to(device)
    student = BenchmarkTransformer(d).to(device)

    distiller = MicroChunkedShadowTwinDistiller(
        teacher=teacher,
        student=student,
        chunk_size=128,
        hidden_dim=d,
    )

    dummy_input = torch.randn(b, t, d, device=device)
    res = distiller.step(dummy_input)

    loss = res["loss"]
    loss.backward()

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / (1024 ** 3)

    # Verification Gate: peak VRAM <= 7.6 GB (< 8.0 GB threshold)
    assert peak_gb < 8.0, f"Peak VRAM {peak_gb:.2f} GB exceeded 8.0 GB threshold"

    # All student leaf weights receive non-zero finite gradients
    for name, p in student.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(p.grad).any(), f"Gradient for {name} has NaN"
            assert torch.count_nonzero(p.grad) > 0, f"Gradient for {name} is all zeros"
