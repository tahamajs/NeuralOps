#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for unified modular surgical delta serialization, loading, and Model Cards."""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file

from neuralops.hub import (
    audit_artifacts_on_disk,
    generate_unified_model_card,
    load_surgeried_model,
    normalize_base_model_id,
    save_surgical_delta,
)
from neuralops.models.surgical_io import extract_surgical_manifest


class _MockSurgeriedAttn(nn.Module):
    is_custom_attention = True
    target_attention = "alibi"

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = dim
        self.num_heads = 2
        self.q_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x))


class _MockAnchorAttn(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)


class _MockCompressedLinear(nn.Module):
    is_compressed = True

    def __init__(self, in_f: int = 16, out_f: int = 16, rank: int = 4) -> None:
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.rank = rank
        self.U = nn.Parameter(torch.randn(out_f, rank))
        self.V = nn.Parameter(torch.randn(rank, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))


class _MockVLMModel(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.vision_tower = nn.Module()
        self.vision_tower.self_attn = nn.Linear(dim, dim)
        self.multi_modal_projector = nn.Linear(dim, dim)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = nn.Embedding(32, dim)
        layer0 = nn.Module()
        layer0.self_attn = _MockSurgeriedAttn(dim)
        layer0.input_layernorm = nn.LayerNorm(dim)
        layer0.mlp = _MockCompressedLinear(dim, dim, rank=4)
        layer1 = nn.Module()
        layer1.self_attn = _MockAnchorAttn(dim)
        layer1.input_layernorm = nn.LayerNorm(dim)
        self.language_model.model.layers = nn.ModuleList([layer0, layer1])


def test_save_surgical_delta_excludes_frozen_vision_and_anchors(tmp_path: Path) -> None:
    model = _MockVLMModel()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.language_model.model.layers[0].self_attn.parameters():
        p.requires_grad = True
    for p in model.language_model.model.layers[0].input_layernorm.parameters():
        p.requires_grad = True
    for p in model.language_model.model.layers[0].mlp.parameters():
        p.requires_grad = True

    stale = tmp_path / "pytorch_model.bin"
    stale.write_bytes(b"legacy-full-weights")

    trained_names = [n for n, p in model.named_parameters() if p.requires_grad]
    summary = save_surgical_delta(
        model,
        tmp_path,
        base_model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        surgeried_prefixes=["language_model.model.layers.0.self_attn"],
        trained_param_names=trained_names,
        anchor_prefixes=["language_model.model.layers.1.self_attn"],
        operations=[{"component_type": "attention", "operator_key": "alibi", "pattern": "hybrid_3_1"}],
    )

    assert not stale.exists(), "Legacy full weights must be purged"
    weights_file = tmp_path / "attention_weights.safetensors"
    manifest_file = tmp_path / "surgical_manifest.json"
    assert weights_file.exists()
    assert manifest_file.exists()

    saved = load_file(str(weights_file))
    assert any(k.startswith("language_model.model.layers.0.self_attn.") for k in saved)
    assert any(k.startswith("language_model.model.layers.0.mlp.") for k in saved)
    assert any(k.startswith("language_model.model.layers.0.input_layernorm.") for k in saved)
    assert not any(k.startswith("vision_tower.") for k in saved), "Frozen vision_tower leaked!"
    assert not any(k.startswith("language_model.model.layers.1.self_attn.") for k in saved), "Anchor leaked!"
    assert not any("embed_tokens" in k for k in saved), "Frozen embed_tokens leaked!"

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "neuralops.surgery/v2"
    assert summary["saved_tensor_count"] == len(saved)
    extracted = extract_surgical_manifest(model)
    assert "language_model.model.layers.0.self_attn" in extracted["submodules"]
    assert "language_model.model.layers.0.mlp" in extracted["submodules"]


def test_unified_model_card_all_six_zoos_honest_audit(tmp_path: Path) -> None:
    (tmp_path / "attention_weights.safetensors").write_bytes(b"0" * 4096)
    (tmp_path / "config.json").write_text('{"attention_type": "cosformer"}', encoding="utf-8")

    for zoo_type in (
        "bert_attention",
        "bert_compressor",
        "qwen_attention",
        "qwen_compressor",
        "llava_vlm",
        "tinyllama",
    ):
        card = generate_unified_model_card(
            zoo_type=zoo_type,
            mechanism_key="cosformer",
            repo_id=f"tahamajs/test-{zoo_type}",
            out_dir=tmp_path,
            metrics={
                "acc_before": 40.0,
                "acc_after": 62.5,
                "ppl_before": 120.0,
                "ppl_after": 22.5,
                "orig_params": 100_000_000,
                "comp_params": 50_000_000,
                "steps_s1": 20,
                "steps_s2": 40,
                "train_secs": 30.0,
                "peak_vram_mb": 12000,
            },
        )
        assert "license: mit" in card
        assert "model-index:" in card
        assert "load_surgeried_model" in card
        assert "from scripts.benchmarks" not in card
        assert "attention_weights.safetensors" in card
        assert "pytorch_model.bin" not in card


def test_unified_model_card_yaml_frontmatter_and_nonfinite_guard(tmp_path: Path) -> None:
    """Verify YAML frontmatter parses cleanly and non-finite metrics are excluded from model-index."""
    (tmp_path / "attention_weights.safetensors").write_bytes(b"0" * 1_048_576)
    (tmp_path / "surgical_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "neuralops.surgery/v2",
                "metadata": {
                    "pattern": "hybrid_3_1",
                    "total_layers": 24,
                    "surgeried_layers": [i for i in range(24) if i % 4 != 3],
                    "anchor_layers": [3, 7, 11, 15, 19, 23],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"attention_type": "retnet", "weights_format": "attention_only_fp16", "saved_tensor_count": 72}),
        encoding="utf-8",
    )

    card = generate_unified_model_card(
        zoo_type="llava_vlm",
        mechanism_key="retnet",
        repo_id="tahamajs/llava-qwen-0.5b-retnet-scienceqa",
        base_model="/home/user/models/llava_qwen_0.5b",
        out_dir=tmp_path,
        metrics={
            "teacher_acc": 72.5,
            "acc_before": 41.0,
            "acc_after": 68.5,
            "ppl_before": float("inf"),
            "ppl_after": float("nan"),
            "steps_s1": 30,
            "steps_s2": 70,
            "train_secs": 120.0,
            "peak_vram_mb": 11500,
        },
    )
    fm = yaml.safe_load(card.split("---\n", 2)[1])
    assert fm["language"] == ["en"]
    assert fm["license"] == "mit"
    assert fm["library_name"] == "neuralops"
    assert fm["base_model"] == "llava-hf/llava-interleave-qwen-0.5b-hf"
    assert fm["pipeline_tag"] == "image-text-to-text"
    assert fm["datasets"] == ["derek-thomas/ScienceQA"]
    assert len(fm["tags"]) == len(set(fm["tags"]))

    idx_metrics = fm["model-index"][0]["results"][0]["metrics"]
    assert [m["type"] for m in idx_metrics] == ["accuracy"]
    assert idx_metrics[0]["value"] == 68.5
    assert "1.00 MB" in card
    assert "`saved_tensor_count=72`" in card
    assert "`schema_version=neuralops.surgery/v2`" in card
    assert "`18 / 24`" in card
    assert "[3, 7, 11, 15, 19, 23]" in card
    assert "~55 MB" not in card
    assert "pytorch_model.bin" not in card


def test_honest_artifact_audit_empty_and_legacy_bin(tmp_path: Path) -> None:
    """Audit must never invent sizes/filenames on empty dirs, and must reflect pytorch_model.bin if present."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_audit = audit_artifacts_on_disk(empty_dir)
    assert empty_audit["audited_dir_exists"] is True
    assert empty_audit["weight_files"] == []
    card_empty = generate_unified_model_card(
        zoo_type="bert_attention",
        mechanism_key="linear_attention",
        repo_id="tahamajs/bert-linear-attention-sst2",
        out_dir=empty_dir,
        metrics={"acc_after": 86.2},
    )
    assert "attention_weights.safetensors" not in card_empty
    assert "pytorch_model.bin" not in card_empty
    assert "~55 MB" not in card_empty

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "pytorch_model.bin").write_bytes(b"x" * 2048)
    card_legacy = generate_unified_model_card(
        zoo_type="bert_attention",
        mechanism_key="linear_attention",
        repo_id="tahamajs/bert-linear-attention-sst2",
        out_dir=legacy_dir,
        metrics={"acc_after": 86.2},
    )
    assert "pytorch_model.bin" in card_legacy
    assert "attention_weights.safetensors" not in card_legacy
    assert normalize_base_model_id("/dev/shm/Qwen2.5-0.5B") == "Qwen/Qwen2.5-0.5B"


def test_vlm_24_layer_and_llama_22_layer_delta_exclusion(tmp_path: Path) -> None:
    """Verify exact 18/24 (VLM) and 17/22 (TinyLlama) hybrid_3_1 delta saving excludes vision & anchors."""
    from scripts.benchmarks.llama_zoo_train_mix import build_llama_readme
    from scripts.benchmarks.vlm_llava_scienceqa_zoo import build_vlm_readme

    class _FullVLM(nn.Module):
        def __init__(self, dim: int = 8) -> None:
            super().__init__()
            self.vision_tower = nn.Module()
            self.vision_tower.vision_model = nn.Module()
            self.vision_tower.vision_model.encoder = nn.Module()
            self.vision_tower.vision_model.encoder.layers = nn.ModuleList([nn.Module() for _ in range(27)])
            for layer in self.vision_tower.vision_model.encoder.layers:
                layer.self_attn = nn.Linear(dim, dim)
            self.multi_modal_projector = nn.Linear(dim, dim)
            self.language_model = nn.Module()
            self.language_model.model = nn.Module()
            self.language_model.model.layers = nn.ModuleList([nn.Module() for _ in range(24)])
            for i, layer in enumerate(self.language_model.model.layers):
                layer.self_attn = _MockAnchorAttn(dim) if i % 4 == 3 else _MockSurgeriedAttn(dim)
                layer.input_layernorm = nn.LayerNorm(dim)
                layer.mlp = nn.Linear(dim, dim)

    vlm = _FullVLM()
    surgeried_vlm = [i for i in range(24) if i % 4 != 3]
    anchor_vlm = [i for i in range(24) if i % 4 == 3]
    surg_pref_vlm = [f"language_model.model.layers.{i}.self_attn" for i in surgeried_vlm]
    anch_pref_vlm = [f"language_model.model.layers.{i}.self_attn" for i in anchor_vlm]
    trained_vlm = [
        n
        for n, _ in vlm.named_parameters()
        if not n.startswith(("vision_tower.", "multi_modal_projector."))
        and (any(n.startswith(p + ".") for p in surg_pref_vlm) or "layernorm" in n.lower())
    ]

    vlm_dir = tmp_path / "vlm_out"
    vlm_info = save_surgical_delta(
        vlm,
        vlm_dir,
        base_model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        surgeried_prefixes=surg_pref_vlm,
        trained_param_names=trained_vlm,
        anchor_prefixes=anch_pref_vlm,
        operations=[{"op": "replace_attention", "attn_key": "alibi", "pattern": "hybrid_3_1"}],
        metadata={"pattern": "hybrid_3_1", "total_layers": 24, "surgeried_layers": surgeried_vlm, "anchor_layers": anchor_vlm},
    )
    vlm_sd = load_file(str(vlm_dir / "attention_weights.safetensors"))
    assert vlm_info["saved_tensor_count"] == len(vlm_sd)
    assert not any(k.startswith(("vision_tower.", "multi_modal_projector.")) for k in vlm_sd)
    for ai in anchor_vlm:
        assert not any(k.startswith(f"language_model.model.layers.{ai}.self_attn.") for k in vlm_sd)
    for si in surgeried_vlm:
        assert any(k.startswith(f"language_model.model.layers.{si}.self_attn.") for k in vlm_sd)

    vlm_card = build_vlm_readme(
        "alibi",
        {"acc_before": 40.0, "acc_after": 69.0, "ppl_after": 7.8, "pattern": "hybrid_3_1", "total_layers": 24},
        72.0,
        6.5,
        vlm_dir,
        {},
    )
    assert "`18 / 24`" in vlm_card
    assert "from scripts.benchmarks" not in vlm_card

    class _FullLlama(nn.Module):
        def __init__(self, dim: int = 8) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(32, dim)
            self.model.layers = nn.ModuleList([nn.Module() for _ in range(22)])
            for i, layer in enumerate(self.model.layers):
                layer.self_attn = _MockAnchorAttn(dim) if i % 4 == 3 else _MockSurgeriedAttn(dim)
                layer.input_layernorm = nn.LayerNorm(dim)
                layer.mlp = nn.Linear(dim, dim)
            self.lm_head = nn.Linear(dim, 32, bias=False)

    llama = _FullLlama()
    surgeried_llama = [i for i in range(22) if i % 4 != 3]
    anchor_llama = [i for i in range(22) if i % 4 == 3]
    surg_pref_llama = [f"model.layers.{i}.self_attn" for i in surgeried_llama]
    anch_pref_llama = [f"model.layers.{i}.self_attn" for i in anchor_llama]
    trained_llama = [
        n
        for n, _ in llama.named_parameters()
        if any(n.startswith(p + ".") for p in surg_pref_llama) or "layernorm" in n.lower()
    ]

    llama_dir = tmp_path / "llama_out"
    save_surgical_delta(
        llama,
        llama_dir,
        base_model_id="TinyLlama/TinyLlama_v1.1",
        surgeried_prefixes=surg_pref_llama,
        trained_param_names=trained_llama,
        anchor_prefixes=anch_pref_llama,
        operations=[{"op": "replace_attention", "attn_key": "alibi", "pattern": "hybrid_3_1"}],
        metadata={"pattern": "hybrid_3_1", "total_layers": 22, "surgeried_layers": surgeried_llama, "anchor_layers": anchor_llama},
    )
    llama_sd = load_file(str(llama_dir / "attention_weights.safetensors"))
    assert not any(k.startswith(("model.embed_tokens.", "lm_head.")) for k in llama_sd)
    assert not any(".mlp." in k for k in llama_sd)
    for ai in anchor_llama:
        assert not any(k.startswith(f"model.layers.{ai}.self_attn.") for k in llama_sd)
    for si in surgeried_llama:
        assert any(k.startswith(f"model.layers.{si}.self_attn.") for k in llama_sd)

    llama_card = build_llama_readme(
        "alibi",
        {"ppl_before": 95.0, "ppl_after": 17.5, "pattern": "hybrid_3_1", "total_layers": 22},
        11.2,
        llama_dir,
        {},
    )
    assert "`17 / 22`" in llama_card
    assert "[3, 7, 11, 15, 19]" in llama_card


def test_qwen_compressor_delta_and_bert_pareto_card(tmp_path: Path) -> None:
    """Verify Qwen compressor delta saving excludes FP16-rounded base embeddings and BERT Pareto card."""
    from scripts.benchmarks.export_bert_compressor_zoo_hf import build_compressor_readme, select_showcase_rank
    from scripts.benchmarks.qwen_compressor_zoo_train_mix import (
        _DELTA_KEYWORDS,
        _build_qwen_comp_readme,
        _save_compressed_weights,
    )

    class _DummyQwen(nn.Module):
        def __init__(self, dtype: torch.dtype = torch.float32) -> None:
            super().__init__()
            self.config = type("Cfg", (), {"num_hidden_layers": 2})()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(32, 8)
            self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            for layer in self.model.layers:
                layer.self_attn = nn.Module()
                layer.self_attn.q_proj = nn.Linear(8, 8)
                layer.mlp = nn.Module()
                layer.mlp.gate_proj = nn.Linear(8, 16)
            self.lm_head = nn.Linear(8, 32, bias=False)
            if dtype == torch.float16:
                self.half()

    torch.manual_seed(0)
    teacher = _DummyQwen(dtype=torch.float16)
    student = _DummyQwen(dtype=torch.float32)
    student.load_state_dict({k: v.float() for k, v in teacher.state_dict().items()})
    # Simulate compressing layer 0 self_attn.q_proj only (replace with factorized module + int8 buffer)
    student.model.layers[0].self_attn.q_proj = _MockCompressedLinear(8, 8, rank=2)
    student.model.layers[0].self_attn.q_proj.register_buffer("qzeros", torch.zeros(2, dtype=torch.int8))

    # Reproduce the FP16-safe comparison from train_single_qwen_compressor
    teacher_sd = teacher.state_dict()
    teacher_shapes = {k: tuple(v.shape) for k, v in teacher_sd.items()}
    comp_prefixes = []
    for k, v in student.state_dict().items():
        t_v = teacher_sd.get(k)
        if t_v is None or tuple(v.shape) != teacher_shapes.get(k):
            changed = True
        elif any(term in k.lower() for term in _DELTA_KEYWORDS):
            changed = True
        elif v.is_floating_point() and t_v.is_floating_point():
            changed = not torch.equal(v.detach().cpu().half(), t_v.detach().cpu().half())
        else:
            changed = v.dtype != t_v.dtype or not torch.equal(v.detach().cpu(), t_v.detach().cpu())
        if changed and "." in k:
            comp_prefixes.append(k.rsplit(".", 1)[0] + ".")

    assert not any(p.startswith(("model.embed_tokens.", "lm_head.")) for p in comp_prefixes)

    qwen_dir = tmp_path / "qwen_comp"
    qwen_dir.mkdir()
    (qwen_dir / "pytorch_model.bin").write_bytes(b"stale")
    cnt, saved_prefs = _save_compressed_weights(
        student,
        qwen_dir,
        comp_key="direct_svd",
        rank=2,
        base_model="/dev/shm/Qwen2.5-0.5B",
        compressed_prefixes=tuple(sorted(set(comp_prefixes))),
        trained_param_names={"model.layers.0.self_attn.q_proj.U", "model.layers.0.self_attn.q_proj.V"},
    )
    assert cnt >= 3
    assert not (qwen_dir / "pytorch_model.bin").exists()
    q_sd = load_file(str(qwen_dir / "attention_weights.safetensors"))
    assert "model.embed_tokens.weight" not in q_sd
    assert "lm_head.weight" not in q_sd
    assert q_sd["model.layers.0.self_attn.q_proj.qzeros"].dtype == torch.int8

    q_card = _build_qwen_comp_readme(
        repo_id="tahamajs/qwen-2.5-0.5b-comp-direct-svd-wikitext2",
        base_model="/dev/shm/Qwen2.5-0.5B",
        comp_key="direct_svd",
        rank=2,
        orig_params=1000,
        comp_params=600,
        param_reduction=1.667,
        sparsity_pct=0.0,
        pre_ppl=32.0,
        post_ppl=19.4,
        steps_s1=10,
        steps_s2=30,
        train_secs=25.0,
        peak_vram_mb=4096,
        out_dir=qwen_dir,
        modified_prefixes=saved_prefs,
    )
    q_fm = yaml.safe_load(q_card.split("---\n", 2)[1])
    assert q_fm["base_model"] == "Qwen/Qwen2.5-0.5B"
    assert any(m["type"] == "compression_ratio" for m in q_fm["model-index"][0]["results"][0]["metrics"])

    ranks = [
        {"rank": 128, "status": "PASSED", "ratio": 1.25, "orig_params": 109483778, "comp_params": 87587022, "zero_shot_acc": 88.0, "healed_acc": 88.0, "lora_healed": False},
        {"rank": 64, "status": "PASSED", "ratio": 1.65, "orig_params": 109483778, "comp_params": 66353804, "zero_shot_acc": 79.1, "healed_acc": 89.2, "lora_healed": True},
    ]
    best = select_showcase_rank(ranks)
    assert best is not None and best["rank"] == 64

    bert_dir = tmp_path / "bert_comp"
    bert_dir.mkdir()
    (bert_dir / "attention_weights.safetensors").write_bytes(b"0" * 8192)
    b_card = build_compressor_readme("direct_svd", best, ranks, {}, bert_dir)
    b_fm = yaml.safe_load(b_card.split("---\n", 2)[1])
    assert b_fm["base_model"] == "google-bert/bert-base-uncased"
    assert "Multi-Rank Pareto Evaluation" in b_card
    assert "⭐" in b_card
    assert "pytorch_model.bin" not in b_card


def test_load_surgeried_model_local_roundtrip(tmp_path: Path) -> None:
    """Verify save_surgical_delta + surgical_manifest.json + config.json round-trips via load_surgeried_model."""
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=2,
    )
    teacher = BertForSequenceClassification(cfg)
    from neuralops.api import replace_attention

    student = replace_attention(BertForSequenceClassification(cfg), attention="cosformer", copy_weights=False)
    for n, p in student.named_parameters():
        p.requires_grad = "attention" in n or "classifier" in n

    save_surgical_delta(
        student,
        tmp_path,
        base_model_id="bert-base-uncased",
        surgeried_prefixes=["bert.encoder.layer.0.attention", "bert.encoder.layer.1.attention", "classifier"],
        trained_param_names=[n for n, p in student.named_parameters() if p.requires_grad],
        operations=[{"op": "replace_attention", "attn_key": "cosformer", "pattern": "all"}],
        metadata={"attention_type": "cosformer", "pattern": "all", "total_layers": 2},
    )
    cfg_dict = cfg.to_dict()
    cfg_dict.update(
        {
            "attention_type": "cosformer",
            "surgery_pattern": "all",
            "weights_format": "attention_only_fp16",
            "base_model_name_or_path": "bert-base-uncased",
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    class _DummyTok:
        pass

    with (
        patch("transformers.BertForSequenceClassification.from_pretrained", return_value=teacher),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_DummyTok()),
    ):
        loaded_model, loaded_tok = load_surgeried_model(str(tmp_path), device="cpu")

    assert isinstance(loaded_tok, _DummyTok)
    x = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        logits = loaded_model(input_ids=x).logits
    assert logits.shape == (1, 2)
    assert math.isfinite(float(logits.abs().max()))
