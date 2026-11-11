#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for offline causal LM and VLM teacher representation caching."""
from __future__ import annotations

import contextlib
import os
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pre-warm modules once at import time
import neuralops.layers  # noqa: F401 - populate ATTENTION_REGISTRY
from scripts.benchmarks.qwen_zoo_train_mix import (
    evaluate_causal_ppl,
    main as qwen_main,
    train_single_qwen_attention,
)
from scripts.benchmarks.llama_zoo_train_mix import (
    main as llama_main,
    train_single_llama_attention,
)
from scripts.benchmarks.qwen_compressor_zoo_train_mix import (
    main as qwen_comp_main,
    train_single_qwen_compressor,
)
from scripts.benchmarks.vlm_llava_scienceqa_zoo import (
    _collate_vlm_batch,
    evaluate_vlm_scienceqa,
    main as vlm_main,
    train_single_vlm_attention,
)
from scripts.benchmarks.teacher_cache_utils import (
    VLM_MAX_CACHE_BYTES,
    CachedCausalDataset,
    build_or_load_causal_teacher_cache,
    build_or_load_vlm_teacher_cache,
    compute_topk_sparse_kd_loss_from_cache,
    resolve_teacher_cache_path,
    select_strided_layer_indices,
    select_vlm_layer_indices,
    try_load_causal_teacher_cache,
    try_load_vlm_teacher_cache,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*torch.cuda.amp.GradScaler.*:FutureWarning"),
    pytest.mark.filterwarnings("ignore:.*CUDA is not available.*:UserWarning"),
]


class _DummyDecoderAttn(nn.Module):
    """Minimal self-attention compatible with QwenAttentionAdapter weight inheritance."""

    def __init__(self, dim: int = 16, num_heads: int = 4, num_kv_heads: int = 2) -> None:
        super().__init__()
        head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        return self.o_proj(self.q_proj(hidden_states)), None


class _DummyDecoderLayer(nn.Module):
    def __init__(self, dim: int = 16, num_heads: int = 4, num_kv_heads: int = 2) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.self_attn = _DummyDecoderAttn(dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + attn_out
        return x + self.mlp(x)


class _DummyCausalBackbone(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 16, num_layers: int = 6) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([_DummyDecoderLayer(dim=dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ):
        h = self.embed_tokens(input_ids)
        hiddens = [h]
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)
            hiddens.append(h)
        h = self.norm(h)
        return types.SimpleNamespace(
            last_hidden_state=h,
            hidden_states=tuple(hiddens) if output_hidden_states else None,
        )


class _DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 16, num_layers: int = 6) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=dim,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=dim // 4,
            vocab_size=vocab_size,
            is_decoder=True,
            model_type="qwen2",
            to_dict=lambda: {
                "hidden_size": dim,
                "num_hidden_layers": num_layers,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": vocab_size,
            },
        )
        self.model = _DummyCausalBackbone(vocab_size=vocab_size, dim=dim, num_layers=num_layers)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return types.SimpleNamespace(
            logits=logits,
            loss=loss,
            hidden_states=out.hidden_states,
        )


class _DummyVLM(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 16, num_layers: int = 6) -> None:
        super().__init__()
        self.vision_tower = nn.Conv2d(3, dim, kernel_size=1)
        self.multi_modal_projector = nn.Linear(dim, dim)
        self.language_model = _DummyCausalLM(vocab_size=vocab_size, dim=dim, num_layers=num_layers)
        self.config = types.SimpleNamespace(
            image_token_index=31,
            vision_feature_select_strategy="full",
            vision_config=types.SimpleNamespace(patch_size=4, image_size=8),
            text_config=self.language_model.config,
            to_dict=lambda: {"model_type": "llava", "num_hidden_layers": num_layers},
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ):
        if pixel_values is not None:
            _ = self.multi_modal_projector(self.vision_tower(pixel_values.to(self.vision_tower.weight.dtype)).mean(dim=(-2, -1)))
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )


class _DummyTokenizer:
    def __init__(self, vocab_size: int = 32) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [2 + (sum(ord(c) for c in text) % max(1, self.vocab_size - 3))]

    def save_pretrained(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _DummyProcessor:
    def __init__(self, vocab_size: int = 32) -> None:
        self.tokenizer = _DummyTokenizer(vocab_size=vocab_size)

    def save_pretrained(self, out_dir: str) -> None:
        self.tokenizer.save_pretrained(out_dir)


def test_layer_selection_and_path_resolution(tmp_path: Path) -> None:
    assert select_strided_layer_indices(24) == [3, 7, 11, 15, 19, 23]
    assert select_strided_layer_indices(22) == [3, 7, 11, 15, 19, 21]

    vlm_layers = select_vlm_layer_indices(num_layers=24, num_samples=256, avg_seq_len=800, hidden_size=1024)
    est_vlm_bytes = 256 * len(vlm_layers) * 800 * 1024 * 2
    assert est_vlm_bytes <= VLM_MAX_CACHE_BYTES
    assert 23 in vlm_layers

    assert resolve_teacher_cache_path("none", tmp_path, "Qwen/Qwen2.5-0.5B", "qwen_causal", 512, 256, 64) is None
    explicit = tmp_path / "custom.pt"
    assert resolve_teacher_cache_path(str(explicit), tmp_path, "Qwen/Qwen2.5-0.5B", "qwen_causal", 512, 256, 64) == explicit
    auto = resolve_teacher_cache_path("auto", tmp_path, "Qwen/Qwen2.5-0.5B", "qwen_causal", 512, 256, 64)
    assert auto is not None
    assert str(auto).endswith(".pt")


def test_causal_teacher_cache_roundtrip_and_zero_teacher_distillation(tmp_path: Path) -> None:
    torch.manual_seed(42)
    train_chunks = torch.randint(0, 30, (8, 12), dtype=torch.long)
    val_chunks = torch.randint(0, 30, (4, 12), dtype=torch.long)
    cache_path = tmp_path / "causal_cache.pt"

    load_calls = 0

    def _load_teacher() -> nn.Module:
        nonlocal load_calls
        load_calls += 1
        return _DummyCausalLM(vocab_size=32, dim=16, num_layers=6)

    cache = build_or_load_causal_teacher_cache(
        cache_path,
        model_id="dummy/qwen",
        normalized_model_id="dummy/qwen",
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        device=torch.device("cpu"),
        load_teacher_fn=_load_teacher,
        evaluate_ppl_fn=evaluate_causal_ppl,
        seq_len=12,
        max_cache_samples=6,
        topk=8,
        batch_size=4,
    )

    assert load_calls == 1
    assert cache_path.exists()
    assert cache["input_ids"].shape == (6, 12)
    assert cache["input_ids"].dtype == torch.int32
    assert cache["attention_mask"].dtype == torch.uint8
    assert cache["hidden_states"].shape == (6, 2, 12, 16)
    assert cache["hidden_states"].dtype == torch.float16
    assert cache["layer_indices"] == [3, 5]
    assert cache["topk_vals"].shape == (6, 12, 8)
    assert cache["topk_vals"].dtype == torch.float16
    assert cache["topk_idx"].shape == (6, 12, 8)
    assert cache["topk_idx"].dtype == torch.int32

    # Second load MUST hit disk cache and NEVER invoke _load_teacher
    def _fail_if_called() -> nn.Module:
        raise AssertionError("Teacher loader must NOT be called when cache exists!")

    reloaded = build_or_load_causal_teacher_cache(
        cache_path,
        model_id="dummy/qwen",
        normalized_model_id="dummy/qwen",
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        device=torch.device("cpu"),
        load_teacher_fn=_fail_if_called,
        evaluate_ppl_fn=evaluate_causal_ppl,
        seq_len=12,
        max_cache_samples=6,
        topk=8,
        batch_size=4,
    )
    assert reloaded["layer_indices"] == [3, 5]
    assert try_load_causal_teacher_cache(cache_path, model_id="dummy/qwen", seq_len=12, num_samples=6, topk=8) is not None

    # Verify student distillation step using CachedCausalDataset + compute_topk_sparse_kd_loss_from_cache
    ds = CachedCausalDataset(reloaded)
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    val_loader = DataLoader(CachedCausalDataset(reloaded), batch_size=2, shuffle=False)
    batch = next(iter(loader))

    student_probe = _DummyCausalLM(vocab_size=32, dim=16, num_layers=6)
    s_out = student_probe(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
    kd_loss = compute_topk_sparse_kd_loss_from_cache(s_out.logits, batch["teacher_topk_vals"], batch["teacher_topk_idx"])
    assert torch.isfinite(kd_loss) and kd_loss.item() >= 0.0
    kd_loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in student_probe.parameters())

    # End-to-end train_single_qwen_attention with teacher_model=None and teacher_cache=reloaded
    out_dir = tmp_path / "qwen_student_out"
    with patch("scripts.benchmarks.qwen_zoo_train_mix.AutoModelForCausalLM.from_pretrained", return_value=_DummyCausalLM(32, 16, 6)):
        res = train_single_qwen_attention(
            attn_key="linear_attention",
            teacher_model=None,
            train_loader=loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            tokenizer=_DummyTokenizer(),
            out_dir=out_dir,
            model_name="dummy/qwen",
            pattern="hybrid_3_1",
            train_secs=0.4,
            topk=8,
            grad_accum=1,
            teacher_cache=reloaded,
        )
    assert res["status"] == "PASSED"
    assert (out_dir / "attention_weights.safetensors").exists()


def test_causal_cache_oom_backoff_during_build(tmp_path: Path) -> None:
    torch.manual_seed(3)
    train_chunks = torch.randint(0, 30, (6, 8), dtype=torch.long)
    val_chunks = torch.randint(0, 30, (2, 8), dtype=torch.long)
    cache_path = tmp_path / "oom_cache.pt"

    teacher = _DummyCausalLM(vocab_size=32, dim=16, num_layers=6)
    orig_forward = teacher.forward

    def _oom_on_large_bs(input_ids: torch.Tensor, **kwargs):
        if input_ids.size(0) > 2:
            raise RuntimeError("CUDA out of memory.")
        return orig_forward(input_ids=input_ids, **kwargs)

    teacher.forward = _oom_on_large_bs  # type: ignore[method-assign]
    cache = build_or_load_causal_teacher_cache(
        cache_path,
        model_id="dummy/oom",
        normalized_model_id="dummy/oom",
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        device=torch.device("cpu"),
        load_teacher_fn=lambda: teacher,
        evaluate_ppl_fn=evaluate_causal_ppl,
        seq_len=8,
        max_cache_samples=6,
        topk=8,
        batch_size=4,
    )
    assert cache["hidden_states"].shape[0] == 6
    assert cache_path.exists()


def test_vlm_teacher_cache_roundtrip_and_zero_teacher_distillation(tmp_path: Path) -> None:
    torch.manual_seed(7)
    train_ex = []
    for i in range(6):
        seq_len = 14 + (i % 4)
        ids = torch.randint(2, 28, (seq_len,), dtype=torch.long)
        labels = ids.clone()
        labels[: seq_len - 2] = -100
        train_ex.append(
            {
                "input_ids": ids,
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "pixel_values": torch.randn(3, 8, 8, dtype=torch.float16),
                "labels": labels,
                "prompt_input_ids": ids[:-2],
                "prompt_attention_mask": torch.ones(seq_len - 2, dtype=torch.long),
                "answer_idx": i % 4,
                "num_choices": 4,
            }
        )
    val_ex = train_ex[:2]
    processor = _DummyProcessor()
    cache_path = tmp_path / "vlm_cache.pt"

    load_calls = 0

    def _load_vlm() -> nn.Module:
        nonlocal load_calls
        load_calls += 1
        return _DummyVLM(vocab_size=32, dim=16, num_layers=6)

    cache = build_or_load_vlm_teacher_cache(
        cache_path,
        model_id="dummy/vlm",
        normalized_model_id="dummy/vlm",
        train_examples=train_ex,
        val_examples=val_ex,
        processor=processor,
        device=torch.device("cpu"),
        load_teacher_fn=_load_vlm,
        evaluate_vlm_fn=evaluate_vlm_scienceqa,
        collate_fn=_collate_vlm_batch,
        max_cache_samples=5,
        topk=8,
        tail_tokens=6,
        batch_size=2,
    )

    assert load_calls == 1
    assert cache_path.exists()
    assert "pixel_values" not in cache
    assert len(cache["hidden_states"]) == 5
    assert cache["hidden_states"][0].dtype == torch.float16
    assert cache["hidden_states"][0].shape == (2, 14, 16)
    assert cache["tail_topk_vals"][0].shape == (6, 8)
    assert cache["tail_topk_idx"][0].dtype == torch.int32

    def _fail_if_called() -> nn.Module:
        raise AssertionError("VLM teacher must NOT be called on cache hit!")

    reloaded = build_or_load_vlm_teacher_cache(
        cache_path,
        model_id="dummy/vlm",
        normalized_model_id="dummy/vlm",
        train_examples=train_ex,
        val_examples=val_ex,
        processor=processor,
        device=torch.device("cpu"),
        load_teacher_fn=_fail_if_called,
        evaluate_vlm_fn=evaluate_vlm_scienceqa,
        collate_fn=_collate_vlm_batch,
        max_cache_samples=5,
        topk=8,
        tail_tokens=6,
        batch_size=2,
    )
    assert try_load_vlm_teacher_cache(cache_path, model_id="dummy/vlm", num_samples=5, topk=8) is not None

    out_dir = tmp_path / "vlm_student_out"
    with patch("transformers.LlavaForConditionalGeneration.from_pretrained", return_value=_DummyVLM(32, 16, 6)):
        res = train_single_vlm_attention(
            attn_key="linear_attention",
            teacher_vlm=None,
            train_examples=train_ex,
            val_examples=val_ex,
            processor=processor,
            device=torch.device("cpu"),
            out_dir=out_dir,
            model_id="dummy/vlm",
            teacher_acc=float(reloaded["teacher_acc"]),
            teacher_ppl=float(reloaded["teacher_ppl"]),
            pattern="hybrid_3_1",
            train_secs=0.4,
            batch_size=2,
            grad_accum=1,
            teacher_cache=reloaded,
        )
    assert res["status"] == "PASSED"
    assert (out_dir / "attention_weights.safetensors").exists()


@pytest.mark.parametrize(
    "name,main_fn",
    [
        ("qwen_zoo_train_mix", qwen_main),
        ("llama_zoo_train_mix", llama_main),
        ("qwen_compressor_zoo_train_mix", qwen_comp_main),
        ("vlm_llava_scienceqa_zoo", vlm_main),
    ],
)
def test_cli_exposes_teacher_cache_flags(name: str, main_fn: Any, capsys: Any) -> None:
    with patch.object(sys, "argv", [f"{name}.py", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main_fn()
        assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--teacher_cache_path" in out
    assert "--max_cache_samples" in out
