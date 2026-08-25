import sys
import os
import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Codes")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from framework.core.registry import COMPRESSOR_REGISTRY, ATTENTION_REGISTRY
from framework.compression_engine.slicegpt import SliceGPTCompressor, slicegpt_factory
from framework.compression_engine.slicegpt_pca import SliceGPTPCACompressor, slicegpt_pca_factory
from framework.compression_engine.sparsegpt import SparseGPTCompressor, sparsegpt_factory
from framework.compression_engine.admm_pruner import ADMMPruner, ADMMMaskedLinear, admm_pruner_factory
from framework.compression_engine.armor_pruner import ARMORPruner, ARMORLinear, armor_pruner_factory
from framework.compression_engine.ast_pruner import ASTPruner, ast_pruner_factory
from framework.compression_engine.bonsai_pruner import BonsaiPruner, bonsai_factory
from framework.compression_engine.flap import FLAPCompressor, flap_compressor_factory
from framework.compression_engine.flat_llm import FLATLLMCompressor, flat_llm_factory
from framework.compression_engine.lillama_compressor import LiLLaMACompressor, lillama_factory
from framework.compression_engine.cachenotes_compressor import (
    CacheNotesCompressor,
    CacheNotesKVCache,
    CompressionPlanningTokens,
)
from framework.compression_engine.come import CoMeCompressor, CoMeBridge, CoMeBridgedLayer, come_factory
from framework.compression_engine.oats_compressor import OATSCompressor, oats_compressor_factory
from framework.compression_engine.proxsparse import ProxSparsePruner, proxsparse_factory
from framework.compression_engine.soka_compressor import LowRankCompressor, LowRankLinear, soka_factory
from framework.compression_engine.liftr import LIFTRCompressor, liftr_factory
from framework.compression_engine.mamba_shedder import MambaShedderCompressor, mamba_shedder_factory
from framework.compression_engine.sdrm_compressor import SDRMCompressor, sdrm_compressor_factory
from framework.attentions.linformer_attention import LinformerAttention
from framework.attentions.adaptive_sparse_attention import AdaptiveSparseAttention
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    compute_empirical_hessian,
    compute_hessian_saliency,
    compute_taylor_first_order_saliency,
    recalibrate_layernorms,
)


class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp_dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.mlp_dense_2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        norm_x = self.ln_1(x)
        q = self.q_proj(norm_x)
        k = self.k_proj(norm_x)
        v = self.v_proj(norm_x)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size), dim=-1)
        attn_out = self.out_proj(torch.matmul(attn, v))
        x = x + attn_out
        norm_x2 = self.ln_2(x)
        mlp_out = self.mlp_dense_2(self.act(self.mlp_dense_1(norm_x2)))
        return x + mlp_out


class ToyModel(nn.Module):
    def __init__(self, num_layers=4, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(hidden_size, intermediate_size)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 100)

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get("input_ids", None)
        if x.dtype == torch.long:
            embed = torch.randn(100, 64, device=x.device)
            x = F.embedding(x, embed)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


class TestTrack5Audits(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_01_slicegpt_orthogonal_invariance(self):
        d_in = 64
        N = 200
        X = torch.randn(N, d_in)
        
        model = ToyModel()
        compressor = SliceGPTCompressor(model, sparsity=0.25, non_uniform=True)
        Q, eigvals = compressor._compute_orthogonal_rotation(X, d_in)
        
        QtQ = torch.matmul(Q.T, Q)
        QQt = torch.matmul(Q, Q.T)
        eye = torch.eye(d_in)
        
        diff_QtQ = torch.norm(QtQ - eye).item()
        diff_QQt = torch.norm(QQt - eye).item()
        
        self.assertLess(diff_QtQ, 1e-4, f"SliceGPT Q^T Q != I: diff = {diff_QtQ}")
        self.assertLess(diff_QQt, 1e-4, f"SliceGPT Q Q^T != I: diff = {diff_QQt}")
        
        comp_model, stats = compressor.compress()
        self.assertIn("sliced_layers", stats)
        self.assertGreater(stats["sliced_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape, (2, 16, 100))

    def test_02_slicegpt_pca(self):
        model = ToyModel()
        compressor = SliceGPTPCACompressor(model, sparsity=0.3)
        comp_model, stats = compressor.compress()
        self.assertIn("sliced_layers", stats)
        self.assertGreater(stats["sliced_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_03_sparsegpt_damped_hessian_and_24_mask(self):
        d_in = 64
        N = 100
        X = torch.randn(N, d_in)
        
        H, H_inv = compute_empirical_hessian(X, damping=0.01)
        self.assertTrue(torch.isfinite(H).all())
        self.assertTrue(torch.isfinite(H_inv).all())
        
        eigvals = torch.linalg.eigvalsh(H)
        self.assertTrue((eigvals > 0).all(), "Empirical Hessian is not positive definite")
        
        prod = torch.matmul(H, H_inv)
        eye = torch.eye(d_in)
        diff = torch.norm(prod - eye).item()
        self.assertLess(diff, 1e-3, f"Hessian inversion error too large: {diff}")
        
        W = torch.randn(32, 64)
        W_24, mask = apply_24_structured_mask(W, dim=1)
        
        W_blocks = W_24.reshape(32, 16, 4)
        nonzeros_per_block = torch.count_nonzero(W_blocks, dim=-1)
        self.assertTrue((nonzeros_per_block == 2).all(), "2:4 mask violated!")
        
        model = ToyModel()
        compressor = SparseGPTCompressor(model, sparsity=0.5, structured_24=True)
        comp_model, stats = compressor.compress()
        self.assertIn("pruned_layers", stats)
        self.assertGreater(stats["pruned_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_04_admm_pruner_convergence_and_ste(self):
        model = ToyModel()
        compressor = ADMMPruner(
            model,
            sparsity=0.5,
            rho=0.1,
            max_iter=6,
            structured_24=True,
            dynamic_rho=True,
            use_ste=True,
            recalibrate_ln=True,
        )
        comp_model, stats = compressor.compress()
        self.assertIn("pruned_layers", stats)
        self.assertGreater(stats["pruned_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64, requires_grad=True)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(dummy_inp.grad)
        self.assertTrue(torch.isfinite(dummy_inp.grad).all())

    def test_05_armor_pruner(self):
        model = ToyModel()
        compressor = ARMORPruner(model, rank=4, sparsity=0.5, structured_24=True, decouple=True)
        comp_model, stats = compressor.compress()
        self.assertIn("pruned_layers", stats)
        self.assertGreater(stats["pruned_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64, requires_grad=True)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(dummy_inp.grad)

    def test_06_ast_pruner(self):
        model = ToyModel()
        compressor = ASTPruner(model, use_pca=True, use_fisher=True, recalibrate_ln=True)
        comp_model, stats = compressor.compress()
        self.assertIn("pruned_layers", stats)
        self.assertGreater(stats["pruned_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_07_bonsai_pruner(self):
        model = ToyModel()
        compressor = BonsaiPruner(
            model,
            sparsity=0.4,
            steps=4,
            recovery_steps=2,
            protect_cls=True,
            recalibrate_ln=True,
        )
        comp_model, stats = compressor.compress()
        self.assertIn("pruned_layers", stats)
        self.assertGreater(stats["pruned_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_08_flap_compressor(self):
        model = ToyModel()
        compressor = FLAPCompressor(model, sparsity=0.25, metric_type="WIFV")
        comp_model, stats = compressor.compress()
        self.assertIn("pruned_layers", stats)
        self.assertGreater(stats["pruned_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_09_flat_llm_compressor(self):
        model = ToyModel()
        compressor = FLATLLMCompressor(model, keep_ratio=0.75, damping=1e-4)
        comp_model, stats = compressor.compress()
        self.assertIn("pruned_layers", stats)
        self.assertGreater(stats["pruned_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_10_lillama_compressor(self):
        model = ToyModel()
        compressor = LiLLaMACompressor(model, rank=16)
        comp_model, stats = compressor.compress()
        self.assertIn("distilled_layers", stats)
        self.assertGreater(stats["distilled_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_11_cachenotes_compressor(self):
        cpt = CompressionPlanningTokens.generate_importance("qa", 64)
        self.assertEqual(cpt.shape, (64,))
        self.assertTrue((cpt > 0).all())
        
        kv_cache = CacheNotesKVCache(max_cache_len=8, task_type="qa", num_sinks=2)
        k_in = torch.randn(2, 4, 16, 32)
        v_in = torch.randn(2, 4, 16, 32)
        k_comp, v_comp = kv_cache.update_and_compress(k_in, v_in)
        self.assertEqual(k_comp.shape, (2, 4, 8, 32))
        self.assertEqual(v_comp.shape, (2, 4, 8, 32))
        
        model = ToyModel()
        compressor = CacheNotesCompressor(model, task_type="reasoning", keep_ratio=0.7)
        comp_model, stats = compressor.compress()
        self.assertIn("compressed_layers", stats)
        self.assertGreater(stats["compressed_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_12_come_compressor(self):
        model = ToyModel(num_layers=4)
        compressor = CoMeCompressor(model, target_layers=2, alpha_init=0.1)
        comp_model, stats = compressor.compress()
        self.assertIn("bridges_inserted", stats)
        self.assertGreaterEqual(stats["bridges_inserted"], 1)
        self.assertEqual(len(comp_model.layers), 2)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_13_oats_compressor(self):
        model = ToyModel()
        compressor = OATSCompressor(model, sparsity=0.5, rank_ratio=0.25, num_iters=3, structured_24=True)
        comp_model, stats = compressor.compress()
        self.assertIn("decomposed_layers", stats)
        self.assertGreater(stats["decomposed_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_14_proxsparse_pruner(self):
        model = ToyModel()
        compressor = ProxSparsePruner(model, sparsity=0.5, lambda_prox=0.01, steps=4, structured_24=True)
        comp_model, stats = compressor.compress()
        self.assertIn("pruned_layers", stats)
        self.assertGreater(stats["pruned_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_15_soka_compressor(self):
        model = ToyModel()
        compressor = LowRankCompressor(model, rank=8)
        comp_model, stats = compressor.compress()
        self.assertIn("total_layers", stats)
        self.assertGreater(stats["total_layers"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_16_liftr_compressor(self):
        model = ToyModel()
        compressor = LIFTRCompressor(model, replace_attention=True)
        comp_model, stats = compressor.compress()
        self.assertIn("linearized_activations", stats)
        self.assertGreater(stats["linearized_activations"], 0)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_17_mamba_shedder(self):
        model = ToyModel(num_layers=4)
        compressor = MambaShedderCompressor(model, sparsity=0.2, shed_layers=1, structured_24=True)
        comp_model, stats = compressor.compress()
        self.assertIn("pruned_layers", stats)
        self.assertGreater(stats["pruned_layers"], 0)
        self.assertEqual(len(comp_model.layers), 3)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_18_sdrm_compressor(self):
        model = ToyModel()
        compressor = SDRMCompressor(model, total_param_ratio=0.6)
        comp_model, stats = compressor.compress()
        self.assertIn("total_param_ratio", stats)
        
        dummy_inp = torch.randn(2, 16, 64)
        out = comp_model(dummy_inp)
        self.assertTrue(torch.isfinite(out).all())

    def test_19_linformer_attention(self):
        class MockConfig:
            num_attention_heads = 4
            hidden_size = 64
            linformer_k = 16
            max_position_embeddings = 128
            dropout = 0.0

        config = MockConfig()
        attn = LinformerAttention(config)
        
        x = torch.randn(2, 32, 64)
        out, att_weights = attn(x)
        self.assertEqual(out.shape, (2, 32, 64))
        self.assertTrue(torch.isfinite(out).all())
        
        mask = torch.ones(2, 32, dtype=torch.long)
        mask[:, 24:] = 0
        out_masked, _ = attn(x, attention_mask=mask)
        self.assertEqual(out_masked.shape, (2, 32, 64))
        self.assertTrue(torch.isfinite(out_masked).all())

    def test_20_adaptive_sparse_attention(self):
        class MockConfig:
            num_attention_heads = 4
            hidden_size = 64
            dropout = 0.0

        config = MockConfig()
        attn = AdaptiveSparseAttention(config, sparsity_ratio=0.5)
        
        x = torch.randn(2, 32, 64)
        out, att_weights = attn(x, output_attentions=True)
        self.assertEqual(out.shape, (2, 32, 64))
        self.assertTrue(torch.isfinite(out).all())
        self.assertIsNotNone(att_weights)
        self.assertEqual(att_weights.shape, (2, 4, 32, 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
