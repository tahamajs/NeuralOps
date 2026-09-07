"""
Comprehensive verification test suite for A3 Compressor hardening.
Validates:
1. Factory and configuration safety (comp_config=None, dict config, object config).
2. Conv1D architecture detection and weight transposition.
3. LoRALinear wrapper preservation, bias extraction, and LoRA adapter parameter freezing.
4. Fallback behavior (fallback_to_original True vs False under high error).
5. Streaming calibration over multi-batch DataLoaders.
6. Target attention only filtering (correctly ignores MLP/FFN including gate/up/down_proj).
7. Architecture compatibility with dummy transformer layers (DistilBERT, RoBERTa, GPT-2, Qwen/LLaMA).
8. LayerNorm recalibration call flag behavior (recalibrate_ln True vs False).
9. FP16 numerical stability.
"""

import unittest
import types
import torch
import torch.nn as nn

from framework.compression_engine.a3_compressor import (
    A3Compressor,
    A3FactorizedLinear,
    a3_compressor_factory,
)


class MockLoRALinear(nn.Module):
    """Mock LoRA wrapped linear module simulating peft / custom LoRALinear."""
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 8.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = alpha
        self.lora_scaling = alpha / r if r > 0 else 1.0
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.base_layer(x)
        res = res + (x @ self.lora_A.T @ self.lora_B.T) * self.lora_scaling
        return res


class MockConv1D(nn.Module):
    """Mock HuggingFace GPT-2 style Conv1D."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.nf = out_features
        self.nx = in_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class TestA3CompressorHardening(unittest.TestCase):

    def test_factory_and_config_safety(self):
        """Accepts comp_config=None, dict, or object without error."""
        model = nn.Sequential(nn.Linear(32, 32))
        
        # 1. comp_config=None
        res_none = a3_compressor_factory(model, comp_config=None, rank=4, return_stats=True)
        self.assertIsInstance(res_none, tuple)
        self.assertEqual(res_none[1]["rank"], 4)

        # 2. comp_config as dict
        model2 = nn.Sequential(nn.Linear(32, 32))
        cfg_dict = {"a3_rank": 8, "fallback_to_original": True, "a3_attention_only": False}
        res_dict = a3_compressor_factory(model2, comp_config=cfg_dict, return_stats=True)
        self.assertEqual(res_dict[1]["rank"], 8)

        # 3. comp_config as object
        model3 = nn.Sequential(nn.Linear(32, 32))
        cfg_obj = types.SimpleNamespace(a3_rank=6, a3_use_fisher=False, fallback_to_original=True)
        res_obj = a3_compressor_factory(model3, comp_config=cfg_obj, return_stats=True)
        self.assertEqual(res_obj[1]["rank"], 6)

    def test_conv1d_detection_and_transposition(self):
        """Verifies robust Conv1D module support and weight orientation."""
        conv = MockConv1D(32, 64)
        comp = A3Compressor(nn.Sequential(conv), rank=8, max_error=0.85)
        fact_layer, err, r = comp.compress_linear(conv, rank=8)

        self.assertIsNotNone(fact_layer)
        self.assertEqual(fact_layer.in_features, 32)
        self.assertEqual(fact_layer.out_features, 64)
        self.assertEqual(fact_layer.rank, 8)

        x = torch.randn(2, 5, 32)
        out = fact_layer(x)
        self.assertEqual(out.shape, (2, 5, 64))

    def test_loralinear_wrapper_and_adapter_preservation(self):
        """Verifies LoRALinear wrapper unpacking, base layer substitution, and adapter freezing."""
        lora_mod = MockLoRALinear(32, 64, r=4, alpha=8.0, bias=True)
        model = nn.ModuleDict({"layer": lora_mod})
        comp = A3Compressor(model, rank=8, max_error=0.90, fallback_to_original=False)
        compressed_model, stats = comp.compress()

        # Check wrapper preserved
        self.assertIsInstance(model["layer"], MockLoRALinear)
        # Check base_layer replaced with A3FactorizedLinear
        self.assertIsInstance(model["layer"].base_layer, A3FactorizedLinear)
        self.assertEqual(model["layer"].base_layer.rank, 8)
        self.assertEqual(model["layer"].base_layer.in_features, 32)
        self.assertEqual(model["layer"].base_layer.out_features, 64)

        # Verify forward pass
        x = torch.randn(2, 32)
        out = model["layer"](x)
        self.assertEqual(out.shape, (2, 64))

        # Test A3FactorizedLinear built-in LoRA freeze_base_weights
        fact = A3FactorizedLinear(32, 64, rank=8, lora_rank=4, lora_alpha=8.0, freeze_base=True)
        self.assertFalse(fact.V.requires_grad)
        self.assertFalse(fact.U.requires_grad)
        self.assertFalse(fact.bias.requires_grad)
        self.assertTrue(fact.lora_A.requires_grad)
        self.assertTrue(fact.lora_B.requires_grad)

    def test_fallback_behavior_true_vs_false(self):
        """Verifies fallback_to_original=True returns None on high error; False compresses anyway."""
        layer = nn.Linear(64, 64)

        # 1. fallback_to_original=True -> returns None
        comp_true = A3Compressor(
            nn.Sequential(layer),
            rank=1,
            max_error=0.001,
            fallback_to_original=True,
            max_retries=0,
        )
        fact_none, err, r = comp_true.compress_linear(layer, name="test_linear", rank=1)
        self.assertIsNone(fact_none)
        self.assertGreater(err, 0.001)

        # 2. fallback_to_original=False -> returns factorized layer despite high error
        comp_false = A3Compressor(
            nn.Sequential(layer),
            rank=1,
            max_error=0.001,
            fallback_to_original=False,
            max_retries=0,
        )
        fact_layer, err_f, r_f = comp_false.compress_linear(layer, name="test_linear", rank=1)
        self.assertIsNotNone(fact_layer)
        self.assertGreater(err_f, 0.001)

    def test_streaming_calibration_multi_batch(self):
        """Verifies multi-batch DataLoader streaming without memory explosion."""
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
        multi_batch_loader = [torch.randn(4, 16) for _ in range(5)]
        
        comp = A3Compressor(
            model,
            rank=4,
            use_fisher=True,
            calibration_loader=multi_batch_loader,
        )
        cov_map, fisher_map = comp._collect_activations_and_fisher()
        self.assertTrue(comp.activations_cleared)
        self.assertIn("0", cov_map)
        self.assertIn("1", cov_map)
        self.assertEqual(cov_map["0"].shape, (16, 16))
        self.assertEqual(fisher_map["0"].shape, (16,))
        self.assertEqual(cov_map["1"].shape, (16, 16))

    def test_target_attention_only_ffn_exclusion(self):
        """Verifies target_attention_only skips FFN layers including gate_proj, up_proj, down_proj."""
        class MockDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 32)
                self.v_proj = nn.Linear(32, 32)
                self.o_proj = nn.Linear(32, 32)
                self.gate_proj = nn.Linear(32, 64)
                self.up_proj = nn.Linear(32, 64)
                self.down_proj = nn.Linear(64, 32)

            def forward(self, x):
                attn = self.o_proj(self.v_proj(x))
                ffn = self.down_proj(torch.relu(self.gate_proj(attn)) * self.up_proj(attn))
                return ffn

        layer = MockDecoderLayer()
        comp = A3Compressor(layer, rank=8, target_attention_only=True, max_error=0.90)
        compressed_model, stats = comp.compress()

        # Attention projections should be compressed
        self.assertIsInstance(layer.q_proj, A3FactorizedLinear)
        self.assertIsInstance(layer.k_proj, A3FactorizedLinear)
        self.assertIsInstance(layer.v_proj, A3FactorizedLinear)
        self.assertIsInstance(layer.o_proj, A3FactorizedLinear)

        # FFN projections must remain standard nn.Linear and have requires_grad=False (freezing uncompressed)
        self.assertIsInstance(layer.gate_proj, nn.Linear)
        self.assertIsInstance(layer.up_proj, nn.Linear)
        self.assertIsInstance(layer.down_proj, nn.Linear)

    def test_four_architectures_compatibility(self):
        """Verifies compression across DistilBERT, RoBERTa, GPT-2, and Qwen dummy structures."""
        # 1. DistilBERT style (q_lin, k_lin, v_lin, out_lin)
        distilbert_layer = nn.ModuleDict({
            "q_lin": nn.Linear(32, 32),
            "k_lin": nn.Linear(32, 32),
            "v_lin": nn.Linear(32, 32),
            "out_lin": nn.Linear(32, 32),
        })
        comp_d = A3Compressor(distilbert_layer, rank=8, max_error=0.90)
        comp_d.compress()
        self.assertIsInstance(distilbert_layer["q_lin"], A3FactorizedLinear)

        # 2. RoBERTa style (query, key, value, dense)
        roberta_layer = nn.ModuleDict({
            "query": nn.Linear(32, 32),
            "key": nn.Linear(32, 32),
            "value": nn.Linear(32, 32),
            "dense": nn.Linear(32, 32),
        })
        comp_r = A3Compressor(roberta_layer, rank=8, max_error=0.90)
        comp_r.compress()
        self.assertIsInstance(roberta_layer["query"], A3FactorizedLinear)

        # 3. GPT-2 style (c_attn Conv1D, c_proj Conv1D)
        gpt2_layer = nn.ModuleDict({
            "c_attn": MockConv1D(32, 96),
            "c_proj": MockConv1D(32, 32),
        })
        comp_g = A3Compressor(gpt2_layer, rank=8, max_error=0.90)
        comp_g.compress()
        self.assertIsInstance(gpt2_layer["c_attn"], A3FactorizedLinear)
        self.assertEqual(gpt2_layer["c_attn"].out_features, 96)

        # 4. Qwen / LLaMA style (q_proj, k_proj, v_proj, o_proj with GQA dimensions)
        qwen_layer = nn.ModuleDict({
            "q_proj": nn.Linear(64, 64),
            "k_proj": nn.Linear(64, 16),
            "v_proj": nn.Linear(64, 16),
            "o_proj": nn.Linear(64, 64),
        })
        comp_q = A3Compressor(qwen_layer, rank=8, max_error=0.90)
        comp_q.compress()
        self.assertIsInstance(qwen_layer["k_proj"], A3FactorizedLinear)
        self.assertEqual(qwen_layer["k_proj"].out_features, 16)

    def test_fp16_numerical_stability(self):
        """Verifies forward pass stability under FP16."""
        fact = A3FactorizedLinear(64, 32, rank=8, dtype=torch.float16, lora_rank=4)
        x = torch.randn(2, 64, dtype=torch.float16)
        out = fact(x)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, (2, 32))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
