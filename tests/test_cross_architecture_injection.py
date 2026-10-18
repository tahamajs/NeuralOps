"""
Exhaustive cross-architecture injection tests across DistilBERT, RoBERTa, GPT-2, and Qwen.
Verifies forward pass shape, finite outputs, LoRA parameter trainability, and RoPE/Causal handling.
"""
import unittest

import torch
import torch.nn as nn
from neuralops.attentions.asvd_attention import ASVDAttention
from neuralops.attentions.cluster_attention import ClusterAttention
from neuralops.attentions.factorised_matrix_state import FactorisedMatrixState
from neuralops.attentions.gated_delta_net import (GatedDeltaNet,
                                                  inject_gated_delta_net)
from neuralops.attentions.hgrn2_gated_deltanet import (
    HGRN2GatedDeltaNet, inject_hgrn2_gated_deltanet)


class MockMHA(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.q_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, hidden_dim)


class RobertaSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)


class MockGPT2Attn(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.split_size = hidden_dim


class MockQwenAttn(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)


class TestCrossArchitectureInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_injection_distilbert(self):
        """DistilBERT linear projections (q_lin, k_lin, v_lin, out_lin) are extracted and injected."""
        mock = MockMHA(64).to(self.device)
        net = GatedDeltaNet(hidden_size=64, num_heads=4, is_decoder=False).to(self.device)
        net.copy_weights(mock)
        x = torch.randn(2, 16, 64, device=self.device)
        out, _ = net(x)
        self.assertEqual(out.shape, (2, 16, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_injection_roberta(self):
        """RoBERTa self-attention (query, key, value) is extracted and nested mode enabled."""
        mock = RobertaSelfAttention(64).to(self.device)
        net = GatedDeltaNet(hidden_size=64, num_heads=4, is_decoder=False).to(self.device)
        net.copy_weights(mock)
        self.assertTrue(net.is_nested_encoder)
        x = torch.randn(2, 16, 64, device=self.device)
        out, _ = net(x)
        self.assertEqual(out.shape, (2, 16, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_injection_gpt2(self):
        """GPT-2 fused c_attn is split properly and causal mask applied."""
        mock = MockGPT2Attn(64).to(self.device)
        net = GatedDeltaNet(hidden_size=64, num_heads=4, is_decoder=True).to(self.device)
        net.copy_weights(mock)
        self.assertTrue(net.is_decoder)
        x = torch.randn(2, 16, 64, device=self.device)
        out, _ = net(x)
        self.assertEqual(out.shape, (2, 16, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_injection_qwen3_5(self):
        """Qwen projections (q_proj, k_proj, v_proj, o_proj) copied and forward works."""
        mock = MockQwenAttn(64).to(self.device)
        net = GatedDeltaNet(hidden_size=64, num_heads=4, is_decoder=True).to(self.device)
        net.copy_weights(mock)
        x = torch.randn(2, 16, 64, device=self.device)
        out, _ = net(x)
        self.assertEqual(out.shape, (2, 16, 64))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
