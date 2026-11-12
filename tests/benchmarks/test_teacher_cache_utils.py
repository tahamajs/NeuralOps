#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for offline causal LM and VLM teacher representation caching utilities."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from scripts.benchmarks.teacher_cache_utils import (
    CachedCausalDataset,
    atomic_torch_save,
    build_or_load_causal_teacher_cache,
    build_or_load_vlm_teacher_cache,
    compute_topk_sparse_kd_loss_from_cache,
    enable_student_gradient_checkpointing,
    resolve_teacher_cache_path,
    select_strided_layer_indices,
    select_vlm_layer_indices,
    try_load_causal_teacher_cache,
    try_load_vlm_teacher_cache,
)


class _DummyCausalTeacher(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16, num_layers: int = 6) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=num_layers, hidden_size=hidden_size)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.oom_above_bs: int | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        if self.oom_above_bs is not None and input_ids.size(0) > self.oom_above_bs:
            raise RuntimeError("CUDA out of memory.")
        h0 = self.emb(input_ids)
        hidden_states = tuple(h0 + (0.1 * i) for i in range(self.num_layers + 1))
        logits = self.proj(hidden_states[-1])
        return SimpleNamespace(hidden_states=hidden_states, logits=logits)


class _DummyVLMTeacher(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16, num_layers: int = 6) -> None:
        super().__init__()
        text_cfg = SimpleNamespace(num_hidden_layers=num_layers, hidden_size=hidden_size)
        self.config = SimpleNamespace(text_config=text_cfg)
        self.language_model = nn.Module()
        self.language_model.config = text_cfg
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.oom_above_bs: int | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        if self.oom_above_bs is not None and input_ids.size(0) > self.oom_above_bs:
            raise RuntimeError("CUDA out of memory.")
        h0 = self.emb(input_ids) + (pixel_values.mean(dim=(1, 2, 3), keepdim=False).view(-1, 1, 1) if pixel_values is not None else 0.0)
        hidden_states = tuple(h0 + (0.05 * i) for i in range(self.num_layers + 1))
        logits = self.proj(hidden_states[-1])
        return SimpleNamespace(hidden_states=hidden_states, logits=logits)


def test_resolve_teacher_cache_path_variants(tmp_path: Path) -> None:
    for off_val in ("none", "OFF", "false", "0"):
        assert resolve_teacher_cache_path(off_val, tmp_path, "Qwen/Qwen2.5-0.5B", "wt2", 512, 256, 64) is None

    explicit = tmp_path / "custom.pt"
    assert resolve_teacher_cache_path(str(explicit), tmp_path, "Qwen/Qwen2.5-0.5B", "wt2", 512, 256, 64) == explicit

    auto_path = resolve_teacher_cache_path("auto", tmp_path, "Qwen/Qwen2.5-0.5B", "wt2", 512, 256, 64)
    assert auto_path is not None


def test_resolve_teacher_cache_path_with_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_dir = tmp_path / "custom_env_cache"
    monkeypatch.setenv("NEURALOPS_CACHE_DIR", str(env_dir))
    resolved = resolve_teacher_cache_path("auto", tmp_path / "out", "Qwen/Qwen2.5-0.5B", "wt2", 512, 256, 64)
    assert resolved == env_dir / "teacher_caches" / "Qwen_Qwen2.5-0.5B_wt2_seq512_n256_k64.pt"


def test_strided_and_vlm_layer_selection() -> None:
    assert select_strided_layer_indices(24) == [3, 7, 11, 15, 19, 23]
    assert select_strided_layer_indices(22) == [3, 7, 11, 15, 19, 21]
    assert select_strided_layer_indices(0) == []

    capped = select_strided_layer_indices(24, max_layers=2)
    assert len(capped) <= 2 and capped[-1] == 23

    # Force VLM byte budget to cap to <= 2 layers
    bytes_per_layer = 100 * 1000 * 1024 * 2
    vlm_idx = select_vlm_layer_indices(24, num_samples=100, avg_seq_len=1000, hidden_size=1024, max_bytes=bytes_per_layer * 2)
    assert len(vlm_idx) <= 2 and vlm_idx[-1] == 23


def test_causal_teacher_cache_build_hit_rebuild_and_oom_backoff(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cache_path = tmp_path / "causal_cache.pt"
    train_chunks = torch.randint(0, 32, (6, 12))
    val_chunks = torch.randint(0, 32, (2, 12))

    teacher = _DummyCausalTeacher(vocab_size=32, hidden_size=16, num_layers=6)
    teacher.oom_above_bs = 2  # Force OOM backoff 4 -> 2
    load_calls = 0

    def _load_teacher() -> nn.Module:
        nonlocal load_calls
        load_calls += 1
        return teacher

    payload = build_or_load_causal_teacher_cache(
        cache_path,
        model_id="Qwen/Qwen2.5-0.5B",
        normalized_model_id="Qwen/Qwen2.5-0.5B",
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        device=torch.device("cpu"),
        load_teacher_fn=_load_teacher,
        evaluate_ppl_fn=lambda m, vl, dev: 12.34,
        seq_len=12,
        max_cache_samples=6,
        topk=8,
        batch_size=4,
    )
    assert load_calls == 1
    assert cache_path.exists()
    assert payload["layer_indices"] == [3, 5]
    assert payload["hidden_states"].shape == (6, 2, 12, 16)
    assert payload["topk_vals"].shape == (6, 12, 8)
    assert payload["topk_idx"].shape == (6, 12, 8)
    assert payload["teacher_ppl"] == pytest.approx(12.34)

    # Cache hit: load_teacher_fn MUST NOT be called again
    def _fail_if_called() -> nn.Module:
        raise AssertionError("load_teacher_fn must not run on cache hit")

    hit_fast = try_load_causal_teacher_cache(
        cache_path, model_id="Qwen/Qwen2.5-0.5B", seq_len=12, num_samples=6, topk=8
    )
    assert hit_fast is not None

    hit_full = build_or_load_causal_teacher_cache(
        cache_path,
        model_id="Qwen/Qwen2.5-0.5B",
        normalized_model_id="Qwen/Qwen2.5-0.5B",
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        device=torch.device("cpu"),
        load_teacher_fn=_fail_if_called,
        evaluate_ppl_fn=lambda m, vl, dev: 99.0,
        seq_len=12,
        max_cache_samples=6,
        topk=8,
    )
    assert hit_full["teacher_ppl"] == pytest.approx(12.34)

    # Dataset + sparse KD loss gradient check
    ds = CachedCausalDataset(hit_full)
    assert len(ds) == 6
    sample = ds[0]
    assert torch.equal(sample["input_ids"], sample["labels"])

    s_logits = torch.randn(2, 12, 32, requires_grad=True)
    kd_loss = compute_topk_sparse_kd_loss_from_cache(
        s_logits, hit_full["topk_vals"][:2], hit_full["topk_idx"][:2], temperature=2.0
    )
    assert torch.isfinite(kd_loss)
    kd_loss.backward()
    assert s_logits.grad is not None and torch.isfinite(s_logits.grad).all()

    # Corrupt file triggers rebuild
    cache_path.write_bytes(b"corrupted-not-torch")
    assert try_load_causal_teacher_cache(cache_path, model_id="Qwen/Qwen2.5-0.5B", seq_len=12, num_samples=6, topk=8) is None
    rebuilt = build_or_load_causal_teacher_cache(
        cache_path,
        model_id="Qwen/Qwen2.5-0.5B",
        normalized_model_id="Qwen/Qwen2.5-0.5B",
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        device=torch.device("cpu"),
        load_teacher_fn=_load_teacher,
        evaluate_ppl_fn=lambda m, vl, dev: 12.34,
        seq_len=12,
        max_cache_samples=6,
        topk=8,
        batch_size=2,
    )
    assert load_calls == 2
    assert rebuilt["num_samples"] == 6


def test_vlm_teacher_cache_build_hit_and_oom_backoff(tmp_path: Path) -> None:
    torch.manual_seed(1)
    cache_path = tmp_path / "vlm_cache.pt"
    train_examples = [
        {"input_ids": torch.randint(0, 32, (10,)), "pixel_values": torch.randn(3, 4, 4)},
        {"input_ids": torch.randint(0, 32, (14,)), "pixel_values": torch.randn(3, 4, 4)},
        {"input_ids": torch.randint(0, 32, (8,)), "pixel_values": torch.randn(3, 4, 4)},
    ]
    val_examples = train_examples[:2]

    def _collate(items, pad_id: int = 0, eval_prompt_only: bool = False):
        max_l = max(int(ex["input_ids"].size(0)) for ex in items)
        in_ids = torch.full((len(items), max_l), pad_id, dtype=torch.long)
        att_m = torch.zeros((len(items), max_l), dtype=torch.long)
        px = torch.stack([ex["pixel_values"] for ex in items], dim=0)
        for i, ex in enumerate(items):
            l_i = int(ex["input_ids"].size(0))
            in_ids[i, :l_i] = ex["input_ids"]
            att_m[i, :l_i] = 1
        return {"input_ids": in_ids, "attention_mask": att_m, "pixel_values": px}

    teacher = _DummyVLMTeacher(vocab_size=32, hidden_size=16, num_layers=6)
    teacher.oom_above_bs = 2
    proc = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

    payload = build_or_load_vlm_teacher_cache(
        cache_path,
        model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        normalized_model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        train_examples=train_examples,
        val_examples=val_examples,
        processor=proc,
        device=torch.device("cpu"),
        load_teacher_fn=lambda: teacher,
        evaluate_vlm_fn=lambda m, v, p, d, batch_size=2: (63.75, 2.45),
        collate_fn=_collate,
        max_cache_samples=3,
        topk=8,
        tail_tokens=6,
        batch_size=3,
    )
    assert payload["teacher_acc"] == pytest.approx(63.75)
    assert payload["seq_lens"] == [10, 14, 8]
    assert payload["tail_start"] == [4, 8, 2]
    assert payload["hidden_states"][0].shape == (2, 10, 16)
    assert payload["tail_topk_vals"][1].shape == (6, 8)

    hit = try_load_vlm_teacher_cache(
        cache_path,
        model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        num_samples=3,
        topk=8,
    )
    assert hit is not None

    # Cache hit skips teacher loading
    hit2 = build_or_load_vlm_teacher_cache(
        cache_path,
        model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        normalized_model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        train_examples=train_examples,
        val_examples=val_examples,
        processor=proc,
        device=torch.device("cpu"),
        load_teacher_fn=lambda: (_ for _ in ()).throw(AssertionError("should not load teacher on hit")),
        evaluate_vlm_fn=lambda *a, **kw: (0.0, 0.0),
        collate_fn=_collate,
        max_cache_samples=3,
        topk=8,
        tail_tokens=6,
    )
    assert hit2["teacher_ppl"] == pytest.approx(2.45)


def test_atomic_save_and_gradient_checkpointing_helper(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "item.pt"
    atomic_torch_save({"x": torch.tensor([1, 2, 3])}, target)
    assert target.exists()
    assert not list(target.parent.glob("*.tmp"))

    lm = MagicMock()
    lm.config = SimpleNamespace(use_cache=True)
    vlm = SimpleNamespace(config=SimpleNamespace(use_cache=True), language_model=lm)
    enable_student_gradient_checkpointing(vlm, log_prefix="[TEST] ")
    assert vlm.config.use_cache is False
    assert lm.config.use_cache is False
    lm.gradient_checkpointing_enable.assert_called_once()
    lm.enable_input_require_grads.assert_called_once()
