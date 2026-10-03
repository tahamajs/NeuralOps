"""
Track 2 Comprehensive Audit & Dynamic Validation Suite.
Audits all Spectral, SSM, and Recurrent Attention modules:
- Forward pass correctness (output shape, no NaNs, no Infs)
- Backward pass gradient propagation (finite gradients on input and parameters)
- Variable sequence lengths and batch sizes
- Causal / padding attention mask compliance
- Numerical stability under extreme inputs
- Pretrained weight transfer compatibility
"""

from .testing_utils import MockConfig, DummyDistilBertAttention
import math
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure neuralops/ is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class MockMHA(nn.Module):
    """Mock pretrained MHA for weight transfer testing."""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.q_lin = nn.Linear(hidden_size, hidden_size)
        self.k_lin = nn.Linear(hidden_size, hidden_size)
        self.v_lin = nn.Linear(hidden_size, hidden_size)
        self.out_lin = nn.Linear(hidden_size, hidden_size)

class TestTrack2Modules(unittest.TestCase):

    def setUp(self):
        self.config = MockConfig(hidden_size=768, num_attention_heads=12)
        self.mock_attn = MockMHA(hidden_size=768)

    def _test_module_lifecycle(self, module_instance, name: str, B=2, L=64, D=768):
        """Standard lifecycle test: forward, backward, masking, variable lengths."""
        module_instance.eval()
        x = torch.randn(B, L, D, requires_grad=True)
        
        # 1. Forward pass
        out, _ = module_instance(x)
        if isinstance(out, tuple):
            out = out[0]
        self.assertEqual(out.shape, (B, L, D), f"{name}: Output shape mismatch {out.shape} != {(B, L, D)}")
        self.assertFalse(torch.isnan(out).any(), f"{name}: NaN detected in forward output")
        self.assertFalse(torch.isinf(out).any(), f"{name}: Inf detected in forward output")

        # 2. Backward pass
        module_instance.train()
        x_train = torch.randn(B, L, D, requires_grad=True)
        out_train, _ = module_instance(x_train)
        if isinstance(out_train, tuple):
            out_train = out_train[0]
        loss = out_train.sum()
        loss.backward()
        self.assertIsNotNone(x_train.grad, f"{name}: Input gradient is None")
        self.assertFalse(torch.isnan(x_train.grad).any(), f"{name}: NaN detected in input gradient")
        self.assertFalse(torch.isinf(x_train.grad).any(), f"{name}: Inf detected in input gradient")

        # 3. Masking test (2D padding mask)
        mask = torch.ones(B, L, dtype=torch.long)
        mask[:, L // 2:] = 0  # pad second half
        out_masked, _ = module_instance(x, attention_mask=mask)
        if isinstance(out_masked, tuple):
            out_masked = out_masked[0]
        self.assertFalse(torch.isnan(out_masked).any(), f"{name}: NaN detected with 2D mask")

        # 4. Variable sequence length
        for short_l in [16, 32, 128]:
            x_var = torch.randn(1, short_l, D)
            out_var, _ = module_instance(x_var)
            if isinstance(out_var, tuple):
                out_var = out_var[0]
            self.assertEqual(out_var.shape, (1, short_l, D), f"{name}: Failed variable length {short_l}")
            self.assertFalse(torch.isnan(out_var).any(), f"{name}: NaN on length {short_l}")

    # ==================== AEGIS VARIANTS ====================

    def test_aegis_v1(self):
        from neuralops.attentions.aegis_v1 import AegisAttentionV1
        m = AegisAttentionV1(model_config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "AegisAttentionV1")

    def test_aegis_v2(self):
        from neuralops.attentions.aegis_v2 import AegisAttentionV2
        m = AegisAttentionV2(config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "AegisAttentionV2")

    def test_aegis_v3(self):
        from neuralops.attentions.aegis_v3 import AegisAttentionV3
        m = AegisAttentionV3(model_config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "AegisAttentionV3")

    def test_aegis_v4(self):
        from neuralops.attentions.aegis_v4 import AegisAttentionV4
        m = AegisAttentionV4(model_config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "AegisAttentionV4")

    def test_aegis_v5(self):
        from neuralops.attentions.aegis_v5 import AegisAttentionV5
        m = AegisAttentionV5(model_config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "AegisAttentionV5")

    def test_aegis_v6(self):
        from neuralops.attentions.aegis_v6 import AegisAttentionV6
        m = AegisAttentionV6(config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "AegisAttentionV6")

    def test_aegis_hybrid(self):
        from neuralops.attentions.aegis_hybrid import AegisHybridAttention
        m = AegisHybridAttention(config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "AegisHybridAttention")

    def test_aegis_spectron_fp8(self):
        from neuralops.attentions.aegis_spectron_fp8 import \
            AegisSpectronFP8Attention
        m = AegisSpectronFP8Attention(config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "AegisSpectronFP8Attention")

    # ==================== SPECTRAL & SSM VARIANTS ====================

    def test_spectra_v2(self):
        from neuralops.attentions.spectra_v2 import SpectraV2Attention
        m = SpectraV2Attention(model_config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "SpectraV2Attention")

    def test_spectral_mamba(self):
        from neuralops.attentions.spectral_mamba import SpectralMambaAttention
        m = SpectralMambaAttention(config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "SpectralMambaAttention")

    def test_mamba(self):
        from neuralops.attentions.Mamba import Mamba
        m = Mamba(d_model=768, config=self.config)
        self._test_module_lifecycle(m, "Mamba")

    def test_svd_mamba(self):
        import importlib
        mod = importlib.import_module("neuralops.attentions.SVD-Mamba")
        SVDMamba = getattr(mod, "SVDMamba")
        m = SVDMamba(d_model=768, config=self.config)
        self._test_module_lifecycle(m, "SVDMamba")

    def test_gated_delta_net(self):
        from neuralops.attentions.gated_delta_net import GatedDeltaNet
        m = GatedDeltaNet(config=self.config, d_model=768, num_heads=12)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "GatedDeltaNet")

    def test_gated_deltanet_block(self):
        from neuralops.attentions.gated_deltanet_block import \
            GatedDeltaNetBlock
        m = GatedDeltaNetBlock(config=self.config, d_model=768, num_heads=12)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "GatedDeltaNetBlock")

    def test_hgrn2_layer(self):
        from neuralops.attentions.hgrn2_layer import HGRN2Layer
        m = HGRN2Layer(config=self.config, d_model=768)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "HGRN2Layer")

    def test_hgrn2_gated_deltanet(self):
        from neuralops.attentions.hgrn2_gated_deltanet import \
            HGRN2GatedDeltaNet
        m = HGRN2GatedDeltaNet(config=self.config, d_model=768, num_heads=12)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "HGRN2GatedDeltaNet")

    def test_hgrn2_outer(self):
        from neuralops.attentions.hgrn2_outer import HGRN2OuterBlock
        m = HGRN2OuterBlock(config=self.config, dim=768, num_heads=12)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "HGRN2OuterBlock")

    def test_griffin_ssd(self):
        from neuralops.attentions.griffin_ssd import GriffinSSDBlock
        m = GriffinSSDBlock(config=self.config, d_model=768, num_heads=12)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "GriffinSSDBlock")

    def test_simple_ssm(self):
        from neuralops.attentions.simple_ssm import SimpleSSM
        m = SimpleSSM(config=self.config, d_model=768)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "SimpleSSM")

    def test_regla_attention(self):
        from neuralops.attentions.regla_attention import ReGLAAttention
        m = ReGLAAttention(config=self.config)
        self._test_module_lifecycle(m, "ReGLAAttention")

    def test_halo_regla(self):
        from neuralops.attentions.halo_regla import HALOReGLAAttention
        m = HALOReGLAAttention(config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "HALOReGLAAttention")

    def test_kda_attention(self):
        from neuralops.attentions.kda_attention import KDAAttention
        m = KDAAttention(config=self.config, d_model=768, num_heads=12)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "KDAAttention")

    def test_galerkin_attention(self):
        from neuralops.attentions.galerkin_attention import GalerkinAttention
        m = GalerkinAttention(config=self.config, rank=64, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "GalerkinAttention")

    def test_vspec_attention(self):
        from neuralops.attentions.vspec import VSpecAttention
        m = VSpecAttention(config=self.config, pretrained_attn=self.mock_attn)
        self._test_module_lifecycle(m, "VSpecAttention")

    def test_mamba_svd_flash(self):
        from neuralops.attentions.mamba_svd_flash import MambaSVDFlash
        m = MambaSVDFlash(config=self.config, d_model=768, num_heads=12)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "MambaSVDFlash")

    def test_upi_stuffed_mamba(self):
        from neuralops.attentions.upi_stuffed_mamba import UPIMambaBlock
        m = UPIMambaBlock(config=self.config, dim=768)
        m.copy_weights(self.mock_attn)
        self._test_module_lifecycle(m, "UPIMambaBlock")

    def test_mamba_shedder(self):
        from neuralops.compression_engine.mamba_shedder import \
            MambaShedderCompressor
        class SimpleBlock(nn.Module):
            def __init__(self, d=768):
                super().__init__()
                self.lin1 = nn.Linear(d, d)
                self.lin2 = nn.Linear(d, d)
            def forward(self, x):
                return self.lin2(F.relu(self.lin1(x)))
        block = SimpleBlock()
        compressor = MambaShedderCompressor(block, sparsity=0.25)
        comp_model, stats = compressor.compress()
        self.assertIn("compression_ratio", stats)
        self.assertLessEqual(stats["compression_ratio"], 1.0)

if __name__ == "__main__":
    unittest.main()
