#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive tests for modular surgical delta IO, universal Hub loader, and zoo routing."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import BertConfig, BertForSequenceClassification

import neuralops.hub as hub_pkg
from neuralops.compression.svd.svd_linear import SVDLinearLayer
from neuralops.core.registry import create_attention
from neuralops.hub import (
    AsyncHFHubUploader,
    apply_decoder_attention_surgery,
    load_pretrained_bert,
    load_pretrained_compressed_bert,
    load_surgeried_model,
    normalize_base_model_id,
    save_surgical_delta,
)
from neuralops.layers.bert_adapter import BertAttentionAdapter
from neuralops.layers.qwen_adapter import QwenAttentionAdapter
from neuralops.models.surgical_io import extract_surgical_manifest, load_surgical_checkpoint


class SimpleLoRALinear(nn.Module):
    """Test double matching scripts/benchmarks/compressor_multirank_lora_eval.py SimpleLoRALinear."""

    def __init__(self, base_layer: nn.Module, r: int = 4, alpha: float = 16.0) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_f = base_layer.in_features
        out_f = base_layer.out_features
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * (1.0 / math.sqrt(r)))
        self.lora_B = nn.Parameter(torch.randn(out_f, r) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.base_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return res + lora_out


class _MockDecoderSelfAttn(nn.Module):
    def __init__(self, dim: int = 32, num_heads: int = 4, num_kv_heads: int = 2) -> None:
        super().__init__()
        head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, None, None]:
        return self.o_proj(self.q_proj(x)), None, None


class _MockDecoderLayer(nn.Module):
    def __init__(self, dim: int = 32, num_heads: int = 4, num_kv_heads: int = 2) -> None:
        super().__init__()
        self.self_attn = _MockDecoderSelfAttn(dim, num_heads, num_kv_heads)
        self.input_layernorm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(self.input_layernorm(x))[0]
        return x + attn_out + self.mlp(x)


class _MockCausalLM(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        dim: int = 32,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        model_type: str = "qwen2",
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type,
            hidden_size=dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=dim // num_heads,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(64, dim)
        self.model.layers = nn.ModuleList(
            [_MockDecoderLayer(dim, num_heads, num_kv_heads) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(dim, 64, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h))


class _MockLLaVAModel(nn.Module):
    def __init__(self, num_layers: int = 4, dim: int = 32, num_heads: int = 4, num_kv_heads: int = 2) -> None:
        super().__init__()
        text_cfg = SimpleNamespace(
            hidden_size=dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=dim // num_heads,
        )
        self.config = SimpleNamespace(model_type="llava", text_config=text_cfg)
        self.vision_tower = nn.Module()
        self.vision_tower.norm = nn.LayerNorm(dim)
        self.vision_tower.proj = nn.Linear(dim, dim)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList(
            [_MockDecoderLayer(dim, num_heads, num_kv_heads) for _ in range(num_layers)]
        )


def test_hub_exports_present() -> None:
    for sym in (
        "load_surgeried_model",
        "apply_decoder_attention_surgery",
        "normalize_base_model_id",
        "save_surgical_delta",
        "load_pretrained_bert",
        "load_pretrained_compressed_bert",
        "AsyncHFHubUploader",
    ):
        assert hasattr(hub_pkg, sym), f"Missing {sym} in neuralops.hub"


def test_extract_surgical_manifest_captures_all_component_types() -> None:
    model = nn.Module()
    model.vision_tower = nn.Module()
    model.vision_tower.norm = nn.LayerNorm(32)
    model.frozen_proj = nn.Linear(32, 32)
    for p in model.parameters():
        p.requires_grad = False

    core_qwen = create_attention("minimal_svd", dim=32, num_heads=4)
    model.qwen_attn = QwenAttentionAdapter(
        core_attention=core_qwen, embed_dim=32, num_heads=4, num_kv_heads=2, head_dim=8
    )

    core_bert = create_attention("alibi", dim=32, num_heads=4)
    model.bert_attn = BertAttentionAdapter(core_attention=core_bert, embed_dim=32, num_heads=4)

    model.lora_proj = SimpleLoRALinear(nn.Linear(32, 32), r=4)
    model.svd_mlp = SVDLinearLayer(nn.Linear(32, 32), rank=8)

    pruned_linear = nn.Linear(32, 32)
    pruned_linear.register_buffer("pruning_mask", torch.ones(32, 32))
    model.pruned_mlp = pruned_linear

    model.trained_norm = nn.LayerNorm(32)
    for p in model.trained_norm.parameters():
        p.requires_grad = True
    model.classifier = nn.Linear(32, 2)
    for p in model.classifier.parameters():
        p.requires_grad = True

    manifest = extract_surgical_manifest(model)
    subs = manifest["submodules"]

    assert subs["qwen_attn"]["category"] == "attention"
    assert subs["qwen_attn"]["wrapper_class_name"] == "QwenAttentionAdapter"
    assert subs["qwen_attn"]["num_kv_heads"] == 2

    assert subs["bert_attn"]["category"] == "attention"
    assert subs["bert_attn"]["wrapper_class_name"] == "BertAttentionAdapter"

    assert subs["lora_proj"]["category"] == "lora_adapter"
    assert subs["lora_proj"]["rank"] == 4

    assert subs["svd_mlp"]["category"] == "compressed_linear"
    assert subs["pruned_mlp"]["category"] == "compressed_linear"
    assert subs["trained_norm"]["category"] == "trained_norm"
    assert subs["classifier"]["category"] == "trained_head"

    assert "vision_tower.norm" not in subs
    assert "frozen_proj" not in subs


def test_save_surgical_delta_and_load_surgical_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    base = nn.Module()
    base.vision_tower = nn.Module()
    base.vision_tower.proj = nn.Linear(32, 32)
    base.embed = nn.Embedding(16, 32)
    base.anchor_attn = nn.Linear(32, 32)
    base.svd_mlp = nn.Linear(32, 32)
    base.lora_proj = nn.Linear(32, 32)
    base.norm = nn.LayerNorm(32)
    base.classifier = nn.Linear(32, 2)

    import copy

    surgeried = copy.deepcopy(base)
    for p in surgeried.parameters():
        p.requires_grad = False

    core_qwen = create_attention("minimal_svd", dim=32, num_heads=4)
    surgeried.qwen_attn = QwenAttentionAdapter(
        core_attention=core_qwen, embed_dim=32, num_heads=4, num_kv_heads=2, head_dim=8
    )
    # Match a fresh base that also has qwen_attn placeholder or anchor
    base.qwen_attn = nn.Linear(32, 32)

    surgeried.svd_mlp = SVDLinearLayer(surgeried.svd_mlp, rank=8)
    surgeried.lora_proj = SimpleLoRALinear(surgeried.lora_proj, r=4)
    for p in surgeried.norm.parameters():
        p.requires_grad = True
    for p in surgeried.classifier.parameters():
        p.requires_grad = True

    # Stale heavy artifacts must be purged
    for stale_name in ("pytorch_model.bin", "model.safetensors", "surgical_model_state.pt", "optimizer.pt"):
        (tmp_path / stale_name).write_bytes(b"stale")

    trained_names = [n for n, p in surgeried.named_parameters() if p.requires_grad]
    summary = save_surgical_delta(
        surgeried,
        tmp_path,
        base_model_id="test/mock-model",
        surgeried_prefixes=["qwen_attn"],
        trained_param_names=trained_names,
        anchor_prefixes=["anchor_attn"],
        operations=[{"component_type": "attention", "operator_key": "minimal_svd"}],
        metadata={"note": "unit-test"},
        half_precision=False,
        remove_full_weights=True,
    )

    for stale_name in ("pytorch_model.bin", "model.safetensors", "surgical_model_state.pt", "optimizer.pt"):
        assert not (tmp_path / stale_name).exists()

    saved = load_file(str(tmp_path / "attention_weights.safetensors"))
    assert summary["saved_tensor_count"] == len(saved)
    assert any(k.startswith("qwen_attn.") for k in saved)
    assert any(k.startswith("svd_mlp.") for k in saved)
    assert any(k.startswith("lora_proj.") for k in saved)
    assert "norm.weight" in saved and "classifier.weight" in saved
    assert not any(k.startswith("vision_tower.") for k in saved)
    assert not any(k.startswith("anchor_attn.") for k in saved)
    assert not any(k.startswith("embed.") for k in saved)

    surgeried.eval()
    reloaded, manifest = load_surgical_checkpoint(base, tmp_path, map_location="cpu", strict=True)
    assert manifest["schema_version"] == "neuralops.surgery/v2"

    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        assert torch.allclose(surgeried.qwen_attn(x)[0], reloaded.qwen_attn(x)[0], atol=1e-5)
        assert torch.allclose(surgeried.svd_mlp(x), reloaded.svd_mlp(x), atol=1e-5)
        assert torch.allclose(surgeried.lora_proj(x), reloaded.lora_proj(x), atol=1e-5)
        assert torch.allclose(surgeried.classifier(surgeried.norm(x)), reloaded.classifier(reloaded.norm(x)), atol=1e-5)


def test_normalize_base_model_id_paths_and_aliases() -> None:
    assert normalize_base_model_id("/dev/shm/Qwen2.5-0.5B") == "Qwen/Qwen2.5-0.5B"
    assert normalize_base_model_id("/dev/shm/Qwen2.5-0.5B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert normalize_base_model_id("/home/user/models/tinyllama_1.1b") == "TinyLlama/TinyLlama_v1.1"
    assert normalize_base_model_id("/home/user/models/llava_qwen_0.5b") == "llava-hf/llava-interleave-qwen-0.5b-hf"
    assert normalize_base_model_id("/home/user/models/bert_sst2_teacher") == "bert-base-uncased"
    assert normalize_base_model_id("google-bert/bert-base-uncased") == "google-bert/bert-base-uncased"


def test_apply_decoder_attention_surgery_qwen_and_llava() -> None:
    qwen = _MockCausalLM(num_layers=4, dim=32, num_heads=4, num_kv_heads=2)
    qwen, targets, anchors = apply_decoder_attention_surgery(qwen, "alibi", pattern="hybrid_3_1", rank=8)
    assert targets == [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
    ]
    assert anchors == ["model.layers.3.self_attn"]
    assert isinstance(qwen.model.layers[0].self_attn, QwenAttentionAdapter)
    assert isinstance(qwen.model.layers[3].self_attn, _MockDecoderSelfAttn)

    out = qwen(torch.randint(0, 64, (2, 8)))
    assert out.logits.shape == (2, 8, 64)
    assert torch.isfinite(out.logits).all()

    llava = _MockLLaVAModel(num_layers=4, dim=32, num_heads=4, num_kv_heads=2)
    llava, vlm_targets, vlm_anchors = apply_decoder_attention_surgery(llava, "alibi", pattern="all", rank=8)
    assert len(vlm_targets) == 4
    assert all(p.startswith("language_model.model.layers.") for p in vlm_targets)
    assert vlm_anchors == []


def test_load_surgeried_model_decoder_attention_end_to_end(tmp_path: Path) -> None:
    torch.manual_seed(1)
    teacher = _MockCausalLM(num_layers=4, dim=32, num_heads=4, num_kv_heads=2)
    fresh_base = _MockCausalLM(num_layers=4, dim=32, num_heads=4, num_kv_heads=2)
    fresh_base.load_state_dict(teacher.state_dict())

    teacher, targets, anchors = apply_decoder_attention_surgery(teacher, "alibi", pattern="hybrid_3_1", rank=8)
    for p in teacher.parameters():
        p.requires_grad = False
    for pref in targets:
        for p in teacher.get_submodule(pref).parameters():
            p.requires_grad = True

    save_surgical_delta(
        teacher,
        tmp_path,
        base_model_id="Qwen/Qwen2.5-0.5B",
        surgeried_prefixes=targets,
        anchor_prefixes=anchors,
        operations=[{"component_type": "attention", "operator_key": "alibi", "pattern": "hybrid_3_1"}],
        half_precision=False,
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "base_model_name_or_path": "Qwen/Qwen2.5-0.5B",
                "attention_type": "alibi",
                "surgery_pattern": "hybrid_3_1",
            }
        ),
        encoding="utf-8",
    )

    dummy_tok = MagicMock()
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fresh_base) as mock_cls,
        patch("transformers.AutoTokenizer.from_pretrained", return_value=dummy_tok),
    ):
        loaded, tok = load_surgeried_model(str(tmp_path), device="cpu")
        mock_cls.assert_called_once()
        assert mock_cls.call_args[0][0] == "Qwen/Qwen2.5-0.5B"
        assert tok is dummy_tok

    x_ids = torch.randint(0, 64, (2, 10))
    teacher.eval()
    with torch.no_grad():
        assert torch.allclose(teacher(x_ids).logits, loaded(x_ids).logits, atol=1e-5)


def test_load_surgeried_model_bert_attention_and_compressed_lora(tmp_path: Path) -> None:
    torch.manual_seed(2)
    cfg = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=2,
    )
    base_bert = BertForSequenceClassification(cfg)
    import copy

    # 1. Compressed + LoRA-healed BERT via attention_weights.safetensors
    comp_dir = tmp_path / "bert_comp"
    comp_student = copy.deepcopy(base_bert)
    from neuralops.core.compression_pipeline import compress_model

    comp_student = compress_model(comp_student, {"compressors": [{"method": "direct_svd", "rank": 8}]})
    # Wrap query linear/SVDLinear in layer 0 with SimpleLoRALinear
    comp_student.bert.encoder.layer[0].attention.self.query = SimpleLoRALinear(
        comp_student.bert.encoder.layer[0].attention.self.query, r=4
    )
    save_surgical_delta(
        comp_student,
        comp_dir,
        base_model_id="google-bert/bert-base-uncased",
        trained_param_names=["classifier.weight", "classifier.bias"],
        operations=[{"op": "compress_model", "method": "direct_svd", "rank": 8}, {"op": "lora", "rank": 4}],
        half_precision=False,
    )
    (comp_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertForSequenceClassification"],
                "base_model_name_or_path": "google-bert/bert-base-uncased",
                "compression_method": "direct_svd",
                "compression_rank": 8,
                "num_labels": 2,
            }
        ),
        encoding="utf-8",
    )

    dummy_tok = MagicMock()
    with (
        patch("transformers.BertForSequenceClassification.from_pretrained", return_value=copy.deepcopy(base_bert)),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=dummy_tok),
    ):
        loaded_comp, _ = load_surgeried_model(str(comp_dir), device="cpu")

    ids = torch.randint(0, 64, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    comp_student.eval()
    with torch.no_grad():
        out_expected = comp_student(input_ids=ids, attention_mask=mask).logits
        out_loaded = loaded_comp(input_ids=ids, attention_mask=mask).logits
        assert torch.allclose(out_expected, out_loaded, atol=1e-5)

    # 2. Legacy pytorch_model.bin fallback when safetensors is absent
    legacy_dir = tmp_path / "bert_legacy"
    legacy_dir.mkdir()
    (legacy_dir / "config.json").write_text((comp_dir / "config.json").read_text(), encoding="utf-8")
    torch.save(comp_student.state_dict(), legacy_dir / "pytorch_model.bin")
    with (
        patch("transformers.BertForSequenceClassification.from_pretrained", return_value=copy.deepcopy(base_bert)),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=dummy_tok),
    ):
        loaded_legacy, _ = load_surgeried_model(str(legacy_dir), device="cpu")
    with torch.no_grad():
        assert torch.allclose(out_expected, loaded_legacy(input_ids=ids, attention_mask=mask).logits, atol=1e-5)

    # 3. BERT attention surgery via replace_attention + save_surgical_delta + load_surgeried_model
    attn_dir = tmp_path / "bert_attn"
    from neuralops.api import replace_attention

    attn_student = replace_attention(copy.deepcopy(base_bert), attention="alibi", copy_weights=False)
    save_surgical_delta(
        attn_student,
        attn_dir,
        base_model_id="google-bert/bert-base-uncased",
        surgeried_prefixes=["bert.encoder.layer.0.attention", "bert.encoder.layer.1.attention"],
        trained_param_names=["classifier.weight", "classifier.bias", "bert.pooler.dense.weight", "bert.pooler.dense.bias"],
        operations=[{"component_type": "attention", "operator_key": "alibi", "pattern": "all"}],
        half_precision=False,
    )
    (attn_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertForSequenceClassification"],
                "base_model_name_or_path": "google-bert/bert-base-uncased",
                "attention_type": "alibi",
                "num_labels": 2,
            }
        ),
        encoding="utf-8",
    )
    with (
        patch("transformers.BertForSequenceClassification.from_pretrained", return_value=copy.deepcopy(base_bert)),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=dummy_tok),
    ):
        loaded_attn, _ = load_surgeried_model(str(attn_dir), device="cpu")
    attn_student.eval()
    with torch.no_grad():
        assert torch.allclose(
            attn_student(input_ids=ids, attention_mask=mask).logits,
            loaded_attn(input_ids=ids, attention_mask=mask).logits,
            atol=1e-5,
        )


def test_zoo_wrappers_route_to_load_surgeried_model() -> None:
    sentinel = (MagicMock(), MagicMock())
    with patch("neuralops.hub.surgery_loader.load_surgeried_model", return_value=sentinel) as mock_loader:
        res_bert = load_pretrained_bert("gated_delta_net", task="sequence-classification", num_labels=2, device="cpu")
        assert res_bert is sentinel
        assert mock_loader.call_args.kwargs["repo_id"] == "tahamajs/bert-gated-delta-net-sst2"

        res_comp = load_pretrained_compressed_bert("direct_svd", device="cpu")
        assert res_comp is sentinel
        assert mock_loader.call_args.kwargs["repo_id"] == "tahamajs/bert-comp-direct-svd-sst2"


def test_async_uploader_zoo_routing_and_delta_cleanup(tmp_path: Path) -> None:
    uploader = AsyncHFHubUploader(token="hf_test_token", cleanup_weights=True)
    assert uploader.hf_org == "tahamajs"

    expected_routes = {
        ("bert_attention_zoo", "gated_delta_net"): "tahamajs/bert-gated-delta-net-sst2",
        ("bert_compressor_zoo_hf", "direct_svd"): "tahamajs/bert-comp-direct-svd-sst2",
        ("qwen_attention_zoo", "gated_delta_net"): "tahamajs/qwen-2.5-0.5b-gated-delta-net-wikitext2",
        ("qwen_compressor_zoo", "direct_svd"): "tahamajs/qwen-2.5-0.5b-comp-direct-svd-wikitext2",
        ("tinyllama_1.1b_zoo", "gated_delta_net"): "tahamajs/tinyllama-1.1b-gated-delta-net-wikitext2",
        ("vlm_llava_scienceqa_zoo", "gated_delta_net"): "tahamajs/llava-qwen-0.5b-gated-delta-net-scienceqa",
    }
    for (zoo_dir, key), expected_repo in expected_routes.items():
        d = tmp_path / zoo_dir / key
        d.mkdir(parents=True, exist_ok=True)
        assert uploader.resolve_repo_id(key, d) == expected_repo

    # Verify config.json + surgical_manifest.json routing even in a generic directory
    generic_dir = tmp_path / "generic_parent" / "alibi"
    generic_dir.mkdir(parents=True)
    (generic_dir / "surgical_manifest.json").write_text('{"zoo_type": "llava_vlm"}', encoding="utf-8")
    assert uploader.resolve_repo_id("alibi", generic_dir) == "tahamajs/llava-qwen-0.5b-alibi-scienceqa"

    # Verify upload strips legacy full weights via ignore_patterns and cleans up all local weights after commit
    stage = tmp_path / "bert_attention_zoo" / "alibi"
    stage.mkdir(parents=True, exist_ok=True)
    for fname in (
        "attention_weights.safetensors",
        "pytorch_model.bin",
        "model.safetensors",
        "surgical_model_state.pt",
        "optimizer.pt",
    ):
        (stage / fname).write_bytes(b"1234")

    mock_api = MagicMock()
    mock_api.upload_folder.return_value = SimpleNamespace(commit_id="abc123")
    with patch("huggingface_hub.HfApi", return_value=mock_api):
        res = uploader._worker_upload_and_cleanup("alibi", stage, {"acc_after": 90.5})

    uploader.shutdown(wait=True)
    assert res["status"] == "UPLOAD_SUCCESS"
    assert res["repo_id"] == "tahamajs/bert-alibi-sst2"
    ignore_pats = mock_api.upload_folder.call_args.kwargs["ignore_patterns"]
    assert "pytorch_model.bin" in ignore_pats
    assert "model.safetensors" in ignore_pats
    for fname in (
        "attention_weights.safetensors",
        "pytorch_model.bin",
        "model.safetensors",
        "surgical_model_state.pt",
        "optimizer.pt",
    ):
        assert not (stage / fname).exists(), f"{fname} should have been cleaned up after upload"
    assert (stage / "upload_success.json").exists()
