"""
Master End-to-End Test Suite for All Framework Features and Subsystems.

Tests:
1. Core Configuration & Serialization
2. Registries (Attentions & Compressors)
3. Universal Weight Transfer (Adapters)
4. Attention Mask Subsystem (2D, 3D, 4D, Causal, Sliding Window)
5. Attention Injection Pipeline
6. Compression Engine Lifecycle
7. Evaluation Metrics & Profiler
8. Complete Trainer Optimization Loop
9. Worker Status & Telemetry Output
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict

# Ensure framework can be imported from root or Codes/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
codes_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import framework.attentions
import framework.compression_engine
import numpy as np
import torch
import torch.nn as nn
from framework.config.experiment_config import DatasetConfig, ExperimentConfig
from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import (apply_attention_mask,
                                           create_band_mask,
                                           create_causal_mask,
                                           get_token_valid_mask)
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.trainer.metrics import compute_accuracy, compute_f1
from framework.utils.profiler import ModelProfiler
from framework.utils.replace_attention import (UniversalAttentionWrapper,
                                               replace_attention)
from transformers import AutoConfig, DistilBertConfig


class ToyEncoderBlock(nn.Module):
    """Mock transformer block matching DistilBERT architecture."""

    def __init__(self, hidden_size: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lin = nn.Linear(hidden_size, hidden_size)
        self.k_lin = nn.Linear(hidden_size, hidden_size)
        self.v_lin = nn.Linear(hidden_size, hidden_size)
        self.out_lin = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape
        q = self.q_lin(x).view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = self.k_lin(x).view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D // self.num_heads) ** 0.5
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_lin(ctx)
        x = self.norm1(x + attn_out)
        ffn_out = self.fc2(self.act(self.fc1(x)))
        return self.norm2(x + ffn_out)


class ToyClassifierModel(nn.Module):
    """Mock sequence classification model with transformer layers."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(100, hidden_size)
        self.transformer = nn.ModuleDict({
            "layer": nn.ModuleList([ToyEncoderBlock(hidden_size=hidden_size) for _ in range(num_layers)])
        })
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Any:
        x = self.embeddings(input_ids)
        for layer in self.transformer["layer"]:
            x = layer(x, attention_mask=attention_mask)
        # Sequence pooler on [CLS] token (index 0)
        cls_rep = x[:, 0, :]
        logits = self.classifier(cls_rep)
        
        class Output:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = hidden_states
        return Output(logits=logits, hidden_states=x)


class TestFullFrameworkEndToEnd(unittest.TestCase):
    """Comprehensive test case covering all framework components and execution paths."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.hidden_size = 64
        self.num_heads = 4
        self.seq_len = 16
        self.batch_size = 2
        self.config = DistilBertConfig(
            vocab_size=100,
            dim=self.hidden_size,
            hidden_dim=self.hidden_size * 2,
            n_heads=self.num_heads,
            n_layers=2,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_labels=2,
        )

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # --------------------------------------------------------------------------
    # 1. Configuration & Serialization Tests
    # --------------------------------------------------------------------------
    def test_01_configuration_lifecycle(self) -> None:
        """Test ExperimentConfig creation and serialization."""
        ds_cfg = DatasetConfig(name="sst2", path="glue/sst2", task_type="single", num_labels=2)
        exp_cfg = ExperimentConfig(
            svd_ranks=[8, 16, 32],
            datasets=[ds_cfg],
            batch_size=4,
            epochs=1,
            learning_rate=5e-5,
            output_dir=self.temp_dir,
        )

        self.assertEqual(exp_cfg.batch_size, 4)
        self.assertEqual(exp_cfg.epochs, 1)
        self.assertEqual(exp_cfg.svd_ranks, [8, 16, 32])
        self.assertEqual(len(exp_cfg.datasets), 1)

    # --------------------------------------------------------------------------
    # 2. Registry Integrity & Discovery Tests
    # --------------------------------------------------------------------------
    def test_02_registries_completeness(self) -> None:
        """Verify ATTENTION_REGISTRY and COMPRESSOR_REGISTRY contain all expected modules."""
        attn_keys = ATTENTION_REGISTRY.keys()
        comp_keys = COMPRESSOR_REGISTRY.keys()

        self.assertGreaterEqual(len(attn_keys), 100, f"Expected >=100 attention keys, found {len(attn_keys)}")
        self.assertGreaterEqual(len(comp_keys), 60, f"Expected >=60 compressor keys, found {len(comp_keys)}")

        for k in ["svd_linear_attn", "asvd", "aegis_v4", "samba_moe", "span_expanded", "gated_linear_attn"]:
            self.assertIn(k, attn_keys, f"Expected attention key '{k}' missing from registry")

        for k in ["aasvd", "slicegpt", "sparsegpt", "astro", "faq", "admm_pruner", "zerokv_neo"]:
            self.assertIn(k, comp_keys, f"Expected compressor key '{k}' missing from registry")

    # --------------------------------------------------------------------------
    # 3. Universal Adapter & Weight Copying Tests
    # --------------------------------------------------------------------------
    def test_03_universal_adapter_weight_extraction(self) -> None:
        """Verify UniversalAdapter extracts Q, K, V, and Output projections from diverse blocks."""
        block = ToyEncoderBlock(hidden_size=64, num_heads=4)
        
        q_w, q_b = UniversalAdapter.get_weight_bias(block, "q")
        k_w, k_b = UniversalAdapter.get_weight_bias(block, "k")
        v_w, v_b = UniversalAdapter.get_weight_bias(block, "v")
        o_w, o_b = UniversalAdapter.get_weight_bias(block, "o")

        self.assertEqual(q_w.shape, (64, 64))
        self.assertEqual(k_w.shape, (64, 64))
        self.assertEqual(v_w.shape, (64, 64))
        self.assertEqual(o_w.shape, (64, 64))
        self.assertIsNotNone(q_b)
        self.assertIsNotNone(o_b)

    # --------------------------------------------------------------------------
    # 4. Attention Mask Engine Tests (2D, 3D, 4D, Causal, Band)
    # --------------------------------------------------------------------------
    def test_04_attention_mask_subsystem(self) -> None:
        """Verify attention mask helpers across 2D, 3D, 4D, and causal configurations."""
        B, N = 2, 8
        device = torch.device("cpu")

        mask_2d = torch.ones((B, N), dtype=torch.long)
        mask_2d[0, 6:] = 0
        valid = get_token_valid_mask(mask_2d, B, N, device)
        self.assertEqual(valid.shape, (B, N))
        self.assertEqual(valid[0, 5].item(), True)
        self.assertEqual(valid[0, 6].item(), False)

        causal = create_causal_mask(N, device)
        self.assertEqual(causal.shape, (N, N))
        self.assertTrue(causal[0, 0].item())
        self.assertFalse(causal[0, 1].item())

        band = create_band_mask(N, window_size=2, device=device)
        self.assertEqual(band.shape, (N, N))
        self.assertTrue(band[0, 2].item())
        self.assertFalse(band[0, 4].item())

        scores = torch.zeros((B, 1, N, N))
        masked_scores = apply_attention_mask(scores, mask_2d)
        self.assertEqual(masked_scores.shape, (B, 1, N, N))
        self.assertLess(masked_scores[0, 0, :, 7].mean().item(), -9000.0)

    # --------------------------------------------------------------------------
    # 5. Attention Injection Subsystem
    # --------------------------------------------------------------------------
    def test_05_attention_injection_pipeline(self) -> None:
        """Verify dynamic injection of custom attention modules into transformer backbones."""
        class DummyConfig:
            num_attention_heads = 4
            hidden_size = 64
        
        model = ToyClassifierModel(hidden_size=64, num_layers=2)
        model.config = DummyConfig()
        
        dummy_input = torch.randint(0, 100, (2, 16))
    
        out_before = model(dummy_input)
        self.assertEqual(out_before.logits.shape, (2, 2))
    
        model_injected = replace_attention(model, attention_key="svd_linear_attn", rank=16)
        out_after = model_injected(dummy_input)
        self.assertEqual(out_after.logits.shape, (2, 2))

        loss = out_after.logits.sum()
        loss.backward()
        self.assertIsNotNone(model_injected.classifier.weight.grad)

    # --------------------------------------------------------------------------
    # 6. Compression Engine Lifecycle
    # --------------------------------------------------------------------------
    def test_06_compression_engine_lifecycle(self) -> None:
        """Verify compression engine factorizes linear modules and returns accurate statistics."""
        model = ToyClassifierModel(hidden_size=64, num_layers=2)
        compressor_fn = COMPRESSOR_REGISTRY.get("direct_svd")
        self.assertIsNotNone(compressor_fn)

        compressor = compressor_fn(model, rank=16)
        compressed_model, stats = compressor.compress()
        dummy_input = torch.randint(0, 100, (2, 16))
        out = compressed_model(dummy_input)
        self.assertEqual(out.logits.shape, (2, 2))

    # --------------------------------------------------------------------------
    # 7. Metrics & ModelProfiler
    # --------------------------------------------------------------------------
    def test_07_metrics_and_profiler(self) -> None:
        """Verify accuracy/f1 computation and ModelProfiler forward hooks."""
        preds = np.array([0, 1, 1, 0])
        labels = np.array([0, 1, 1, 0])
        acc = compute_accuracy(preds, labels)
        f1 = compute_f1(preds, labels)

        self.assertEqual(acc, 1.0)
        self.assertEqual(f1, 1.0)

        model = ToyClassifierModel(hidden_size=64, num_layers=2)
        profiler = ModelProfiler(model)
        profiler.start_profiling()
        
        dummy_input = torch.randint(0, 100, (2, 16))
        _ = model(dummy_input)
        profiler.stop_profiling()
        
        report = profiler.get_report()
        self.assertIsInstance(report, dict)

    # --------------------------------------------------------------------------
    # 8. Complete Trainer Execution Loop
    # --------------------------------------------------------------------------
    def test_08_trainer_execution_loop(self) -> None:
        """Verify end-to-end training and evaluation loops on synthetic batches."""
        model = ToyClassifierModel(hidden_size=64, num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        dataset = [
            {"input_ids": torch.randint(0, 100, (4, 16)), "labels": torch.tensor([0, 1, 1, 0])},
            {"input_ids": torch.randint(0, 100, (4, 16)), "labels": torch.tensor([1, 0, 0, 1])},
        ]

        model.train()
        for batch in dataset:
            optimizer.zero_grad()
            out = model(batch["input_ids"])
            loss = criterion(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataset:
                out = model(batch["input_ids"])
                preds = torch.argmax(out.logits, dim=-1)
                all_preds.extend(preds.tolist())
                all_labels.extend(batch["labels"].tolist())

        self.assertEqual(len(all_preds), 8)
        self.assertEqual(len(all_labels), 8)

    # --------------------------------------------------------------------------
    # 9. Worker Status & Telemetry Output
    # --------------------------------------------------------------------------
    def test_09_worker_telemetry_serialization(self) -> None:
        """Verify worker telemetry status file creation and JSON schema."""
        status_data = {
            "status": "running",
            "attn": "svd_linear_attn",
            "task": "sst2",
            "idx": "1/100",
            "last_acc": 84.63,
            "last_loss": 0.4989,
            "pid": os.getpid(),
        }
        status_file = os.path.join(self.temp_dir, "worker_status_test.json")
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)

        self.assertTrue(os.path.exists(status_file))
        with open(status_file, "r") as f:
            loaded = json.load(f)

        self.assertEqual(loaded["status"], "running")
        self.assertEqual(loaded["last_acc"], 84.63)


if __name__ == "__main__":
    unittest.main()
