#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Verification suite for Agent 5 OOM chunking and isolated trainer hardening."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

import neuralops.layers  # noqa: F401 - populate ATTENTION_REGISTRY
from neuralops.core.registry import create_attention

OOM_CHUNKED_KEYS = [
    "positive_kernel_linear_attention",
    "split_head_hybrid",
    "spectron_fp8",
    "gated_hybrid_kernel_attention",
    "hybrid_attention",
    "hybrid_local_kernel",
    "granite_hybrid",
    "regla_attention",
    "svd_mamba",
    "hymba_transx",
    "jamba_info",
    "samba_m3",
    "samba_moe",
    "upi_stuffed_mamba",
    "vig_attention",
    "nasc_attention",
]


def _make_config(dim: int = 128, num_heads: int = 4, num_kv_heads: int = 2, is_decoder: bool = True):
    return types.SimpleNamespace(
        hidden_size=dim,
        num_attention_heads=num_heads,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=dim // num_heads,
        is_decoder=is_decoder,
        model_type="qwen2" if is_decoder else "bert",
    )


class _MaxTensorTracker(TorchDispatchMode):
    """Tracks the single largest tensor allocation (in elements) during execution."""

    def __init__(self) -> None:
        super().__init__()
        self.max_numel = 0
        self.max_shape: tuple[int, ...] = ()

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._inspect(out)
        return out

    def _inspect(self, obj) -> None:
        if isinstance(obj, torch.Tensor):
            n = obj.numel()
            if n > self.max_numel:
                self.max_numel = n
                self.max_shape = tuple(obj.shape)
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                self._inspect(item)


@pytest.mark.parametrize("attn_key", OOM_CHUNKED_KEYS)
@pytest.mark.parametrize("is_decoder", [True, False])
def test_chunked_modules_forward_backward_both_modes(attn_key: str, is_decoder: bool):
    """All 16 chunked modules must pass forward+backward across chunk boundaries (L=130) in causal & bidirectional modes."""
    torch.manual_seed(7)
    B, L, D, H = 2, 130, 128, 4
    cfg = _make_config(dim=D, num_heads=H, num_kv_heads=2 if is_decoder else H, is_decoder=is_decoder)
    mod = create_attention(attn_key, config=cfg, is_decoder=is_decoder)
    mod.train()

    x = torch.randn(B, L, D, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, -5:] = 0

    out = mod(x, attention_mask=mask)
    y = out[0] if isinstance(out, tuple) else out
    assert y.shape == (B, L, D), f"{attn_key} ({is_decoder=}): unexpected shape {y.shape}"
    assert torch.isfinite(y).all(), f"{attn_key} ({is_decoder=}): non-finite output"

    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    nz_grads = sum(
        1 for p in mod.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert nz_grads > 0, f"{attn_key} ({is_decoder=}): zero parameter gradients"


@pytest.mark.parametrize("attn_key", OOM_CHUNKED_KEYS)
def test_chunked_modules_strict_causality_across_chunk_boundary(attn_key: str):
    """Perturbing tokens >= 70 must not alter causal outputs on tokens < 64 (cross-chunk causal integrity)."""
    torch.manual_seed(11)
    B, L, D, H = 1, 130, 128, 4
    cfg = _make_config(dim=D, num_heads=H, num_kv_heads=2, is_decoder=True)
    mod = create_attention(attn_key, config=cfg, is_decoder=True)
    mod.eval()

    x1 = torch.randn(B, L, D)
    x2 = x1.clone()
    x2[:, 70:, :] = torch.randn(B, L - 70, D) * 5.0

    with torch.no_grad():
        y1 = mod(x1)[0]
        y2 = mod(x2)[0]

    prefix_diff = (y1[:, :64, :] - y2[:, :64, :]).abs().max().item()
    assert prefix_diff < 1e-4, f"{attn_key}: causal leak across chunk boundary (diff={prefix_diff:.2e})"


@pytest.mark.parametrize("attn_key", OOM_CHUNKED_KEYS)
def test_no_unbounded_l2_or_5d_tensor_spikes(attn_key: str):
    """Ensure no module allocates 5D (B, H, L, d, d) or O(B*H*L^2) attention matrix spikes at L=384."""
    torch.manual_seed(3)
    B, L, D, H = 2, 384, 256, 4
    d = D // H  # 64
    cfg = _make_config(dim=D, num_heads=H, num_kv_heads=2, is_decoder=True)
    mod = create_attention(attn_key, config=cfg, is_decoder=True)
    mod.train()

    x = torch.randn(B, L, D, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)

    naive_l2_numel = B * H * L * L          # 1,179,648
    naive_5d_numel = B * H * L * d * d      # 12,582,912

    with _MaxTensorTracker() as tracker:
        y = mod(x, attention_mask=mask)[0]
        y.pow(2).mean().backward()

    assert tracker.max_numel < naive_5d_numel, (
        f"{attn_key}: allocated 5D-scale tensor {tracker.max_shape} ({tracker.max_numel} >= {naive_5d_numel})"
    )
    # On CPU, PyTorch's F.scaled_dot_product_attention math fallback creates (B*H, L, L),
    # and external svd_mamba_fused_pytorch_ref creates (B, L, D, d_state=16); verify all other
    # chunked modules stay strictly below naive_l2_numel.
    if attn_key not in ("hybrid_attention", "svd_mamba"):
        assert tracker.max_numel < naive_l2_numel, (
            f"{attn_key}: allocated L^2-scale tensor {tracker.max_shape} ({tracker.max_numel} >= {naive_l2_numel})"
        )


def test_hybrid_attention_manual_chunked_fallback_bounds_memory():
    """Verify HybridAttention manual attention fallback (when SDPA is unavailable) chunks below L^2."""
    torch.manual_seed(5)
    B, L, D, H = 2, 256, 256, 4
    cfg = _make_config(dim=D, num_heads=H, num_kv_heads=2, is_decoder=True)
    mod = create_attention("hybrid_attention", config=cfg, is_decoder=True)
    mod.train()

    x = torch.randn(B, L, D, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)
    naive_l2_numel = B * H * L * L

    with (
        patch("torch.nn.functional.scaled_dot_product_attention", side_effect=RuntimeError("mock no sdpa")),
        _MaxTensorTracker() as tracker,
    ):
        y = mod(x, attention_mask=mask)[0]
        y.pow(2).mean().backward()

    assert tracker.max_numel < naive_l2_numel, (
        f"hybrid_attention manual fallback allocated {tracker.max_shape} ({tracker.max_numel} >= {naive_l2_numel})"
    )


def test_chunked_dual_form_mathematical_equivalence():
    """Verify chunked dual-form linear attention matches naive 5D/4D cumsum within float32 precision."""
    from neuralops.layers.hybrid.positive_kernel_linear_attention import PositiveKernelLinearAttention
    from neuralops.layers.hybrid.spectron_fp8_attention import _causal_global_linear_attention
    from neuralops.layers.hybrid.split_head_hybrid_attention import causal_kernel_linear_attention

    torch.manual_seed(0)
    B, H, N, F_dim, d = 2, 4, 150, 32, 32
    Q = torch.rand(B, H, N, F_dim) + 0.1
    K = torch.rand(B, H, N, F_dim) + 0.1
    V = torch.randn(B, H, N, d)

    KV_cumsum = (K.unsqueeze(-1) * V.unsqueeze(-2)).cumsum(dim=2)
    K_cumsum = K.cumsum(dim=2)
    ref = torch.matmul(Q.unsqueeze(-2), KV_cumsum).squeeze(-2) / (Q * K_cumsum).sum(dim=-1, keepdim=True).clamp(
        min=1e-5
    )

    out_pkla = PositiveKernelLinearAttention._causal_linear_attention(Q, K, V, past_len=0, chunk_size=64)
    assert (ref - out_pkla).abs().max().item() < 1e-5

    Q_dec = torch.rand(B, H, 80, F_dim) + 0.1
    ref_dec = torch.matmul(Q_dec.unsqueeze(-2), KV_cumsum[:, :, 70:]).squeeze(-2) / (
        Q_dec * K_cumsum[:, :, 70:]
    ).sum(dim=-1, keepdim=True).clamp(min=1e-5)
    out_pkla_dec = PositiveKernelLinearAttention._causal_linear_attention(Q_dec, K, V, past_len=70, chunk_size=64)
    assert (ref_dec - out_pkla_dec).abs().max().item() < 1e-5

    out_shh = causal_kernel_linear_attention(Q, K, V, chunk_size=64)
    assert (ref - out_shh).abs().max().item() < 1e-5

    out_shh_dec = causal_kernel_linear_attention(Q_dec, K, V, chunk_size=64)
    assert (ref_dec - out_shh_dec).abs().max().item() < 1e-5

    Qf = torch.rand(B, N, 32) + 0.1
    Kf = torch.rand(B, N, 32) + 0.1
    Vf = torch.randn(B, N, 64)
    kv_4d = torch.einsum("bnr,bnd->bnrd", Kf, Vf).cumsum(dim=1)
    k_4d = Kf.cumsum(dim=1)
    ref_spec = torch.einsum("bnr,bnrd->bnd", Qf, kv_4d) / torch.einsum("bnr,bnr->bn", Qf, k_4d).unsqueeze(-1).clamp(
        min=1e-6
    )
    out_spec = _causal_global_linear_attention(Qf, Kf, Vf, chunk_size=64)
    assert (ref_spec - out_spec).abs().max().item() < 1e-5

    from neuralops.layers.linear.nasc_attention import NASCAttention
    from neuralops.ops.triton.linear_attn import fused_nasc_spectral_linear_scan_pytorch

    cfg_nasc = _make_config(dim=128, num_heads=4, num_kv_heads=4, is_decoder=True)
    nasc_mod = NASCAttention(cfg_nasc, is_decoder=True)
    Qn = torch.randn(2, 4, 140, 32)
    Kn = torch.randn(2, 4, 140, 32)
    Vn = torch.randn(2, 4, 140, 32)
    vm = torch.ones(2, 140, dtype=torch.bool)
    vm[:, -7:] = False
    for causal_flag in (True, False):
        ref_n = fused_nasc_spectral_linear_scan_pytorch(
            Qn, Kn, Vn, nasc_mod.rff_proj, valid_mask=vm, is_causal=causal_flag
        )
        got_n = nasc_mod._global_spectral_rff_attention(Qn, Kn, Vn, valid_mask=vm, is_causal=causal_flag)
        assert (ref_n - got_n).abs().max().item() < 1e-5


@pytest.mark.parametrize(
    "attn_key",
    [
        "positive_kernel_linear_attention",
        "split_head_hybrid",
        "spectron_fp8",
        "gated_hybrid_kernel_attention",
        "hybrid_attention",
        "hybrid_local_kernel",
        "granite_hybrid",
        "hymba_transx",
        "jamba_info",
        "samba_m3",
        "vig_attention",
    ],
)
def test_kv_cache_decode_step_after_chunked_prefill(attn_key: str):
    """Ensure KV-cache decode (L=1 with past_len=70) runs cleanly on modules supporting past_key_value."""
    torch.manual_seed(19)
    B, L_pre, D, H = 1, 70, 128, 4
    cfg = _make_config(dim=D, num_heads=H, num_kv_heads=2, is_decoder=True)
    mod = create_attention(attn_key, config=cfg, is_decoder=True)
    mod.eval()

    x_pre = torch.randn(B, L_pre, D)
    with torch.no_grad():
        res_pre = mod(x_pre, use_cache=True)
        assert torch.isfinite(res_pre[0]).all()
        past_kv = res_pre[1]
        if past_kv is not None:
            x_step = torch.randn(B, 1, D)
            res_step = mod(x_step, past_key_value=past_kv, use_cache=True)
            assert res_step[0].shape == (B, 1, D)
            assert torch.isfinite(res_step[0]).all()


def test_bert_zoo_isolated_workers_default_and_oom_retry(tmp_path: Path):
    """Verify bert_zoo_train_cached defaults to isolated workers=1, accepts --isolated_workers, and retries OOM down to bs=1."""
    from scripts.benchmarks import bert_zoo_train_cached as bert_mod

    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps(["k1"]), encoding="utf-8")
    metrics_log = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "out"

    # 1. Verify default isolated worker subprocess spawning with --isolated_workers omitted
    spawned_cmds = []

    def _fake_subproc(cmd, *args, **kwargs):
        spawned_cmds.append(cmd)
        return types.SimpleNamespace(returncode=1)

    with (
        patch.object(sys, "argv", [
            "bert_zoo_train_cached.py",
            "--keys_json", str(keys_file),
            "--out_dir", str(out_dir),
            "--metrics_log", str(metrics_log),
            "--device", "cpu",
        ]),
        patch("subprocess.run", side_effect=_fake_subproc),
    ):
        bert_mod.main()

    assert len(spawned_cmds) == 1
    assert "--single_key" in spawned_cmds[0] and "k1" in spawned_cmds[0]
    logged = [json.loads(line) for line in metrics_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert logged[-1]["status"] == "FAIL_CRASH"

    # 2. Verify --single_key retries OOM bs=4 -> bs=2 -> bs=1 with scaled grad_accum and --isolated_workers alias
    attempts = []

    def _fake_train(**kwargs):
        attempts.append((kwargs["batch_size"], kwargs["grad_accum"]))
        if kwargs["batch_size"] > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 896.00 MiB")
        return {"attn_key": kwargs["attn_key"], "status": "PASSED"}

    fake_cache = {
        "input_ids": torch.ones(8, 16, dtype=torch.long),
        "attention_mask": torch.ones(8, 16, dtype=torch.long),
        "labels": torch.zeros(8, dtype=torch.long),
        "teacher_last_hidden": torch.zeros(8, 16, 32),
        "teacher_cls_logits": torch.zeros(8, 2),
    }
    fake_tok = MagicMock()
    fake_tok.return_value = {"input_ids": torch.ones(4, 16, dtype=torch.long), "attention_mask": torch.ones(4, 16, dtype=torch.long)}
    fake_val = {"sentence": ["ok"] * 4, "label": [0, 1, 0, 1]}
    metrics_log_single = tmp_path / "metrics_single.jsonl"

    with (
        patch.object(sys, "argv", [
            "bert_zoo_train_cached.py",
            "--single_key", "k_oom",
            "--isolated_workers", "1",
            "--batch_size", "4",
            "--grad_accum", "2",
            "--out_dir", str(out_dir),
            "--metrics_log", str(metrics_log_single),
            "--device", "cpu",
        ]),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok),
        patch("torch.load", return_value=fake_cache),
        patch("datasets.load_dataset", return_value=fake_val),
        patch.object(bert_mod, "train_single_attention_cached", side_effect=_fake_train),
    ):
        bert_mod.main()

    assert attempts == [(4, 2), (2, 4), (1, 8)]
    rec = json.loads(metrics_log_single.read_text(encoding="utf-8").strip())
    assert rec["status"] == "PASSED"


def test_qwen_zoo_oom_retry_down_to_bs1_with_grad_checkpointing(tmp_path: Path):
    """Verify qwen_zoo_train_mix retries OOM down to bs=1 with scaled grad_accum and gradient_checkpointing=True."""
    from scripts.benchmarks import qwen_zoo_train_mix as qwen_mod

    metrics_log = tmp_path / "qwen_metrics.jsonl"
    out_dir = tmp_path / "qwen_out"
    attempts = []

    def _fake_qwen_train(**kwargs):
        attempts.append((kwargs["batch_size"] if "batch_size" in kwargs else kwargs["train_loader"].batch_size, kwargs["grad_accum"], kwargs["gradient_checkpointing"]))
        if kwargs["train_loader"].batch_size > 1:
            raise RuntimeError("CUDA out of memory.")
        return {"attn_key": kwargs["attn_key"], "status": "PASSED"}

    fake_tok = MagicMock()
    fake_teacher = nn.Linear(4, 4)
    fake_chunks = torch.ones(16, 8, dtype=torch.long)

    with (
        patch.object(sys, "argv", [
            "qwen_zoo_train_mix.py",
            "--attn_key", "positive_kernel_linear_attention",
            "--batch_size", "4",
            "--grad_accum", "2",
            "--out_dir", str(out_dir),
            "--metrics_log", str(metrics_log),
            "--device", "cpu",
        ]),
        patch.object(qwen_mod.AutoTokenizer, "from_pretrained", return_value=fake_tok),
        patch.object(qwen_mod.AutoModelForCausalLM, "from_pretrained", return_value=fake_teacher),
        patch.object(qwen_mod, "load_tokenized_qwen_corpus", return_value=(fake_chunks, fake_chunks)),
        patch.object(qwen_mod, "train_single_qwen_attention", side_effect=_fake_qwen_train),
    ):
        qwen_mod.main()

    assert attempts == [(4, 2, False), (2, 4, True), (1, 8, True)]
    rec = json.loads(metrics_log.read_text(encoding="utf-8").strip())
    assert rec["status"] == "PASSED"


def test_qwen_oom_exhausted_at_bs1_records_skipped_oom(tmp_path: Path):
    """Verify qwen_zoo_train_mix records SKIPPED_OOM only after bs=1 with checkpointing also OOMs."""
    from scripts.benchmarks import qwen_zoo_train_mix as qwen_mod

    metrics_log = tmp_path / "qwen_oom_metrics.jsonl"
    out_dir = tmp_path / "qwen_oom_out"
    attempts = []

    def _always_oom(**kwargs):
        attempts.append((kwargs["train_loader"].batch_size, kwargs["grad_accum"], kwargs["gradient_checkpointing"]))
        raise RuntimeError("CUDA out of memory.")

    fake_tok = MagicMock()
    fake_teacher = nn.Linear(4, 4)
    fake_chunks = torch.ones(16, 8, dtype=torch.long)

    with (
        patch.object(sys, "argv", [
            "qwen_zoo_train_mix.py",
            "--attn_key", "positive_kernel_linear_attention",
            "--batch_size", "2",
            "--grad_accum", "4",
            "--gradient_checkpointing",
            "--out_dir", str(out_dir),
            "--metrics_log", str(metrics_log),
            "--device", "cpu",
        ]),
        patch.object(qwen_mod.AutoTokenizer, "from_pretrained", return_value=fake_tok),
        patch.object(qwen_mod.AutoModelForCausalLM, "from_pretrained", return_value=fake_teacher),
        patch.object(qwen_mod, "load_tokenized_qwen_corpus", return_value=(fake_chunks, fake_chunks)),
        patch.object(qwen_mod, "train_single_qwen_attention", side_effect=_always_oom),
    ):
        qwen_mod.main()

    assert attempts == [(2, 4, True), (1, 8, True)]
    rec = json.loads(metrics_log.read_text(encoding="utf-8").strip())
    assert rec["status"] == "SKIPPED_OOM"
