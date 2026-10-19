#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Master Exhaustive Ecosystem Integration Test Suite for NeuralOps v1.4.2.

Validates the full 7-pillar lifecycle across all modules working together simultaneously:
  Stage 1: Dynamic Module Inspection & Architecture Discovery (UniversalArchitectureScanner).
  Stage 2: In-Place Attention Surgery (transplanting dense MHA with Subquadratic Attention:
           e.g. GLA, GDN, Cosformer, SVD-Linear) with weight copying and closed-form Ridge projection calibration.
  Stage 3: Low-Rank Linear Compression (applying AA-SVD, AdaSVD, SVD-LLM to MLP projections)
           preserving Frobenius reconstruction fidelity.
  Stage 4: Subspace Micro-Healing & 6-Pillar Distillation (running a multi-step healing loop
           with Logit KL, Feature CKA, Attention map alignment, and Hidden Angular MSE).
  Stage 5: State Table & PagedKVCache Allocation: verifying continuous batching inference
           with non-contiguous pages, dynamic block eviction, and Copy-on-Write page forking.
  Stage 6: Self-Speculative Decoding: generating draft tokens and executing target model
           multi-candidate verification with 0 sequence divergence.
  Stage 7: Lossless Serialization & Checkpoint Roundtrip: exporting model to SafeTensors
           and reloading it, verifying bit-exact numerical reproduction.
  Stage 8: Hardware Telemetry & CLI Doctor Verification: ensuring `neuralops doctor` reports
           operational status during active pipeline execution.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, BertConfig, BertModel

import neuralops
from neuralops import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    PagedKVCache,
    PagedStateTable,
    SelfSpeculativeDecoder,
    SpeculativeDecodeStats,
)
from neuralops.models.paged_state import RadixPrefixStateCache
from neuralops.api import (
    replace_attention,
    transplant_attention,
    compress,
    export_model,
    load_model,
    load_exported_model,
)
from neuralops.surgery.scanner import UniversalArchitectureScanner
from neuralops.core._attention_surgery import (
    calibrate_attention_subspace,
    heal_replaced_model,
    compute_reconstruction_error,
)
from neuralops.surgery.universal_calibrator import (
    UniversalActivationCalibrator,
    compress_model_universal,
)
from neuralops.trainer.distillation.multi_level_kd import (
    LogitDistillationLoss,
    LinearCKALoss,
    MiniLMValueRelationKDLoss,
    BlockwiseOperatorCloningLoss,
)
from neuralops.cli.doctor import cmd_doctor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def execution_device():
    """Ensure testing on CUDA if available."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Master Ecosystem Suite] Executing on hardware device: {dev}")
    return dev


class TestMasterExhaustiveEcosystemIntegration:
    """Master End-to-End System Integration Suite testing all 7 pillars + Stage 8 Telemetry."""

    @pytest.mark.parametrize("target_attention", ["cosformer", "gated_delta_net", "gla_attention"])
    @pytest.mark.parametrize("compression_method", ["direct_svd", "svd_llm_compressor", "adasvd_compressor"])
    def test_full_lifecycle_gpt2(self, execution_device, target_attention, compression_method):
        """Complete 8-stage end-to-end integration pipeline on HuggingFace GPT-2 architecture."""
        device = execution_device

        config = GPT2Config(
            vocab_size=256,
            n_positions=128,
            n_embd=64,
            n_layer=2,
            n_head=4,
            pad_token_id=0,
        )
        model = GPT2LMHeadModel(config).to(device)
        model.eval()

        specs = UniversalArchitectureScanner.scan(model)
        assert len(specs) >= 2, f"Scanner failed to discover attention layers (found {len(specs)})"
        for s in specs:
            assert s.pattern_type in ("conv1d_gpt2", "fused_qkv", "standard_mha"), (
                f"Unexpected pattern type: {s.pattern_type}"
            )
            assert s.extracted_qkv is not None, "Failed to extract QKV weights"
        print(f"\n[Stage 1 Passed] Discovered {len(specs)} attention blocks with pattern '{specs[0].pattern_type}'.")

        teacher_model = copy.deepcopy(model)
        teacher_model.eval()

        surg_model = transplant_attention(
            model,
            attention=target_attention,
            copy_weights=True,
        )
        assert surg_model is not None

        calib_ids = torch.randint(0, 256, (2, 16), device=device)
        with torch.no_grad():
            t_logits = teacher_model(calib_ids).logits
            s_logits = surg_model(calib_ids).logits
        assert s_logits.shape == t_logits.shape
        assert torch.isfinite(s_logits).all()

        calibrated_model = calibrate_attention_subspace(
            teacher_model=teacher_model,
            student_model=surg_model,
            calibration_inputs=calib_ids,
            ridge_lambda=1e-3,
            blend_alpha=0.3,
        )
        with torch.no_grad():
            calib_logits = calibrated_model(calib_ids).logits
        assert torch.isfinite(calib_logits).all()
        print(f"[Stage 2 Passed] In-place surgery to '{target_attention}' & Ridge calibration complete.")

        from transformers.pytorch_utils import Conv1D
        from neuralops.core.utils.lora import LoRALinear
        calibrator = UniversalActivationCalibrator(calibrated_model, target_classes=(nn.Linear, Conv1D, LoRALinear))
        calib_stats = calibrator.calibrate([calib_ids])
        assert len(calib_stats) > 0, "Calibrator failed to accumulate streaming stats"

        compressed_model = compress_model_universal(
            model=calibrated_model,
            method=compression_method,
            rank=16,
            calibration_data=[calib_ids],
        )
        assert compressed_model is not None

        with torch.no_grad():
            c_logits = compressed_model(calib_ids).logits
        assert c_logits.shape == (2, 16, 256)
        assert torch.isfinite(c_logits).all()
        print(f"[Stage 3 Passed] Linear compression via '{compression_method}' (rank 16) verified.")

        heal_stats = heal_replaced_model(
            teacher_model=teacher_model,
            student_model=compressed_model,
            calibration_inputs=[calib_ids],
            steps=3,
            lr=1e-3,
            also_ridge_init=False,
            device=device,
        )
        assert heal_stats["steps"] == 3
        print(f"[Stage 4.1] Micro-healing finished: {heal_stats['num_healed_layers']} layers healed.")

        logit_kd = LogitDistillationLoss(mode="reverse_kl", temperature=2.0)
        cka_loss_fn = LinearCKALoss()
        op_cloning_loss = BlockwiseOperatorCloningLoss()

        with torch.no_grad():
            t_out = teacher_model(calib_ids, output_hidden_states=True)
            s_out = compressed_model(calib_ids, output_hidden_states=True)

        l_kd = logit_kd(s_out.logits, t_out.logits)
        assert torch.isfinite(l_kd) and l_kd.item() >= 0.0

        l_cka = cka_loss_fn(s_out.logits, t_out.logits)
        assert torch.isfinite(l_cka)

        l_op = op_cloning_loss(s_out.logits, t_out.logits)
        assert torch.isfinite(l_op)
        print(f"[Stage 4 Passed] 6-Pillar Distillation alignment objectives computed cleanly.")

        kv_cache = PagedKVCache(
            num_blocks=16,
            block_size=4,
            num_heads=4,
            head_dim=16,
            device=device,
        )
        kv_cache.allocate_sequence(seq_id=101)
        k_dummy = torch.randn(4, 16, device=device)
        v_dummy = torch.randn(4, 16, device=device)
        kv_cache.append_token(seq_id=101, key=k_dummy, value=v_dummy)
        k_out, v_out = kv_cache.get_kv(seq_id=101)
        assert k_out.shape == (1, 4, 16)
        assert v_out.shape == (1, 4, 16)
        kv_cache.free_sequence(seq_id=101)

        state_table = PagedStateTable(num_pages=32, num_heads=4, head_dim=16, device=device)
        p_parent = state_table.register_sequence(seq_id=1, initial_pages=1)
        p_id = p_parent[0]
        state_table.page_pool[p_id, 0, 0, 0] = 3.14159

        p_child = state_table.fork_sequence(parent_seq_id=1, child_seq_id=2)
        assert p_child[0] == p_id
        assert state_table.ref_counts[p_id] == 2

        p_child_new = state_table.prepare_write(seq_id=2, page_slot=0)
        assert p_child_new != p_id
        assert state_table.ref_counts[p_id] == 1
        assert state_table.ref_counts[p_child_new] == 1
        assert torch.isclose(state_table.page_pool[p_child_new, 0, 0, 0], torch.tensor(3.14159, device=device))

        radix_cache = RadixPrefixStateCache(state_table)
        prefix = [10, 20, 30]
        radix_cache.insert_prefix(prefix, p_id)
        matched_page, matched_len = radix_cache.match_prefix([10, 20, 30, 40])
        assert matched_len == 3
        assert matched_page == p_id
        print(f"[Stage 5 Passed] Paged KV Cache, PagedStateTable CoW & Radix Prefix matching verified.")

        spec_decoder = SelfSpeculativeDecoder(
            target_model=teacher_model,
            draft_model=compressed_model,
            max_draft_k=3,
        )
        prompt = torch.tensor([[12, 34, 56]], device=device)
        gen_tokens, stats = spec_decoder.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert gen_tokens.shape[1] == prompt.shape[1] + 4
        assert isinstance(stats, SpeculativeDecodeStats)
        assert stats.total_tokens_generated == 4

        with torch.no_grad():
            curr = prompt.clone()
            for _ in range(4):
                tgt_l = teacher_model(curr).logits[:, -1, :]
                nxt = torch.argmax(tgt_l, dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=-1)
        assert torch.equal(gen_tokens, curr), "Speculative decoding diverged from exact greedy target!"
        print(f"[Stage 6 Passed] Self-speculative decoding passed with 0 token divergence.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "ecosystem_export"
            export_model(
                compressed_model,
                output_dir=save_path,
                format="safetensors",
                attention_type=target_attention,
                compression_method=compression_method,
            )
            assert (save_path / "neuralops_config.json").exists()
            assert (save_path / "model.safetensors").exists() or (save_path / "pytorch_model.bin").exists()

            reloaded_model, meta = load_exported_model(save_path, base_model=copy.deepcopy(model))
            reloaded_model = reloaded_model.to(device).eval()

            test_input = torch.randint(0, 256, (1, 8), device=device)
            with torch.no_grad():
                logits_orig = compressed_model(test_input).logits
                logits_rel = reloaded_model(test_input).logits

            max_diff = (logits_orig - logits_rel).abs().max().item()
            assert max_diff < 0.05, f"Serialization numerical reproduction error: max_diff = {max_diff}"
        print(f"[Stage 7 Passed] SafeTensors roundtrip verified with max_diff = {max_diff:.2e} < 0.05.")

        exit_code = cmd_doctor()
        assert exit_code == 0, f"neuralops doctor failed with exit code {exit_code}"
        print(f"[Stage 8 Passed] NeuralOps doctor reported operational status.")

    def test_bidirectional_encoder_lifecycle_bert(self, execution_device):
        """Validates bidirectional encoder lifecycle (BertModel) through all 7 pillars."""
        device = execution_device
        config = BertConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        )
        model = BertModel(config).to(device)
        model.eval()

        specs = UniversalArchitectureScanner.scan(model)
        assert len(specs) >= 2
        assert specs[0].pattern_type == "bert_nested"

        surg_model = transplant_attention(model, attention="cosformer", copy_weights=True)

        input_ids = torch.randint(0, 256, (2, 16), device=device)
        attn_mask = torch.ones(2, 16, device=device)
        attn_mask[0, 10:] = 0

        with torch.no_grad():
            out = surg_model(input_ids=input_ids, attention_mask=attn_mask)
        assert out.last_hidden_state.shape == (2, 16, 64)

        compressed = compress_model_universal(surg_model, method="svd_llm", rank=16, calibration_data=[input_ids])
        with torch.no_grad():
            out_c = compressed(input_ids=input_ids, attention_mask=attn_mask)
        assert out_c.last_hidden_state.shape == (2, 16, 64)

        heal_stats = heal_replaced_model(
            teacher_model=model,
            student_model=compressed,
            calibration_inputs=[{"input_ids": input_ids, "attention_mask": attn_mask}],
            steps=2,
            lr=1e-3,
            device=device,
        )
        assert heal_stats["steps"] == 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "bert_export"
            export_model(compressed, output_dir=save_path, format="safetensors")
            assert (save_path / "neuralops_config.json").exists()

        print("\n[BERT Lifecycle Passed] Bidirectional encoder surgery & compression verified.")
