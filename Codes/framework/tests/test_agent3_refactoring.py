"""
Comprehensive Unit Tests for Agent 3 Refactoring:
1. FP16 Variance Underflow / FP32 Accumulators in RMSNorm and LayerNorm (layer_norm_fix.py).
2. Zipfian Power-Law Token Sampling and Perplexity Dataset (synthetic_generator.py).
3. Tikhonov Damped Matrix Square Root & 2-Wasserstein Hessian Drift (detectors.py).
4. Streaming Layerwise Analysis OOM Prevention (layerwise_analyzer.py).
5. Grassmann / Riemannian Manifold SVD Basis Federated Averaging (grassmann_fedavg.py).
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn
# 4. Layerwise Analyzer
from framework.analysis.layerwise_analyzer import (
    LayerwiseAnalyzer, StreamingLayerAccumulator, estimate_intrinsic_dimension)
# 2. Synthetic Generator
from framework.data.synthetic_generator import (SyntheticClassificationDataset,
                                                SyntheticPerplexityDataset,
                                                compute_empirical_entropy,
                                                generate_zipfian_markov_tokens,
                                                generate_zipfian_tokens)
from framework.drift.config import DriftConfig
# 3. Drift & Wasserstein-2
from framework.drift.detectors import (METHOD_REGISTRY, DriftDetector,
                                       bures_wasserstein_distance,
                                       hessian_wasserstein, matrix_square_root,
                                       wasserstein)
from framework.federated_learning.config import AggregationStrategy, FLConfig
from framework.federated_learning.strategies import \
    AGGREGATOR_REGISTRY as FL_REGISTRY
# 5. Federated Learning Grassmann
from framework.federated_learning.strategies.grassmann_fedavg import (
    GrassmannFedAvgAggregator, grassmann_exp_map,
    grassmann_frechet_mean_spectral, grassmann_karcher_mean, grassmann_log_map,
    is_orthogonal_basis_tensor)
# 1. LayerNorm / RMSNorm FP32 Accumulator
from framework.utils.layer_norm_fix import (LayerNorm, RMSNorm, SafeLayerNorm,
                                            SafeRMSNorm, fix_layernorm_naming,
                                            replace_layernorm_with_safe_norm)


class TestAgent3Refactoring(unittest.TestCase):
    """Test suite for mathematical and statistical stability refactorings."""

    # -------------------------------------------------------------------------
    # 1. FP16 Accumulator & LayerNorm / RMSNorm Tests
    # -------------------------------------------------------------------------

    def test_rmsnorm_fp16_large_activation_stability(self):
        """Test RMSNorm with extreme values that would overflow FP16 square if uncast."""
        norm = RMSNorm(normalized_shape=128, eps=1e-6)
        # 300^2 = 90,000 > 65,504 (FP16 overflow boundary)
        x_fp16 = torch.full((4, 16, 128), 300.0, dtype=torch.float16)
        out = norm(x_fp16)
        self.assertEqual(out.dtype, torch.float16)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        # For constant tensor, rms = 300, x / rms = 1.0
        self.assertTrue(torch.allclose(out, torch.ones_like(out), atol=1e-2))

    def test_layernorm_fp16_stability(self):
        """Test LayerNorm with FP32 accumulator on mixed-precision tensors."""
        norm = LayerNorm(normalized_shape=64, eps=1e-5)
        x = torch.randn(2, 8, 64, dtype=torch.float16) * 50.0 + 200.0
        out = norm(x)
        self.assertEqual(out.dtype, torch.float16)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_replace_layernorm_with_safe_norm(self):
        """Test recursive replacement of standard nn.LayerNorm."""
        model = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.Linear(32, 10),
        )
        replace_layernorm_with_safe_norm(model)
        self.assertIsInstance(model[1], LayerNorm)

    # -------------------------------------------------------------------------
    # 2. Zipfian Token Generator Tests
    # -------------------------------------------------------------------------

    def test_zipfian_token_distribution(self):
        """Verify Zipfian power-law rank frequency decay."""
        tokens = generate_zipfian_tokens(
            num_samples=100, seq_len=500, vocab_size=1000, zipf_param=1.2, seed=42
        )
        self.assertEqual(tokens.shape, (100, 500))
        self.assertEqual(tokens.dtype, torch.long)
        self.assertTrue((tokens >= 1).all())
        self.assertTrue((tokens <= 1000).all())

        # Low-rank tokens should appear much more frequently than high-rank tokens
        flat = tokens.view(-1).numpy()
        counts = np.bincount(flat, minlength=1001)
        freq_top10 = np.sum(counts[1:11])
        freq_tail = np.sum(counts[500:510])
        self.assertGreater(freq_top10, freq_tail * 5)

        # Check empirical entropy
        entropy = compute_empirical_entropy(tokens, vocab_size=1000)
        self.assertGreater(entropy, 2.0)
        self.assertLess(entropy, 10.0)

    def test_synthetic_perplexity_dataset(self):
        """Verify SyntheticPerplexityDataset provides causal LM targets."""
        ds = SyntheticPerplexityDataset(num_samples=16, seq_len=64, vocab_size=5000)
        self.assertEqual(len(ds), 16)
        item = ds[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertEqual(item["input_ids"].shape, (64,))
        self.assertTrue(torch.equal(item["input_ids"], item["labels"]))

    # -------------------------------------------------------------------------
    # 3. Drift & Wasserstein-2 Hessian Stability Tests
    # -------------------------------------------------------------------------

    def test_matrix_square_root_with_negative_eigenvalues(self):
        """Test matrix_square_root regularizes indefinite/ill-conditioned matrices."""
        # Create an indefinite symmetric matrix with negative eigenvalues
        A = np.array([
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, -0.5],
        ])
        sqrt_A = matrix_square_root(A, damping=1e-5, min_eig=1e-8)
        self.assertEqual(sqrt_A.shape, (3, 3))
        self.assertFalse(np.isnan(sqrt_A).any())
        self.assertFalse(np.isinf(sqrt_A).any())
        # Square of square root should be positive semi-definite
        reconstructed = sqrt_A @ sqrt_A
        eigs = np.linalg.eigvalsh(reconstructed)
        self.assertTrue(np.all(eigs >= 0.0))

    def test_bures_wasserstein_distance_properties(self):
        """Verify Bures 2-Wasserstein metric properties (identity, symmetry, positivity)."""
        cov1 = np.array([[2.0, 0.5], [0.5, 1.5]])
        cov2 = np.array([[3.0, 0.2], [0.2, 2.0]])
        mean1 = np.array([0.0, 1.0])
        mean2 = np.array([1.0, 0.0])

        # Distance to self must be 0
        w_self = bures_wasserstein_distance(cov1, cov1, mean1, mean1)
        self.assertAlmostEqual(w_self, 0.0, places=4)

        # Symmetry: W(A, B) == W(B, A)
        w_12 = bures_wasserstein_distance(cov1, cov2, mean1, mean2)
        w_21 = bures_wasserstein_distance(cov2, cov1, mean2, mean1)
        self.assertAlmostEqual(w_12, w_21, places=5)
        self.assertGreater(w_12, 0.0)

    def test_drift_detector_hessian_wasserstein(self):
        """Verify DriftDetector integration with registered hessian_wasserstein method."""
        ref_data = {"feat1": np.random.randn(100), "feat2": np.random.randn(100)}
        cur_data = {"feat1": np.random.randn(100) + 2.0, "feat2": np.random.randn(100) * 3.0}
        config = DriftConfig(methods=["ks", "hessian_wasserstein"], threshold=0.1)
        detector = DriftDetector(ref_data, config=config)
        report = detector.detect(cur_data)
        self.assertIsNotNone(report)
        self.assertTrue(len(report.features) == 2)
        self.assertIn("hessian_wasserstein", report.features[0].methods)

    # -------------------------------------------------------------------------
    # 4. Streaming Layerwise Analysis OOM Prevention Tests
    # -------------------------------------------------------------------------

    def test_streaming_layer_accumulator(self):
        """Test streaming accumulator updates moments on-the-fly."""
        acc = StreamingLayerAccumulator()
        t1 = torch.randn(4, 16, 64)
        t2 = torch.randn(4, 16, 64)
        acc.update(t1)
        acc.update(t2)
        self.assertEqual(acc.total_tokens, 128)
        self.assertEqual(acc.running_mean_vector.shape, (64,))
        self.assertGreater(acc.max_activation, 0.0)
        self.assertGreaterEqual(acc.compute_effective_rank(), 1.0)
        self.assertGreaterEqual(acc.compute_outlier_ratio(), 0.0)

    def test_layerwise_analyzer_streaming(self):
        """Test LayerwiseAnalyzer on mock transformer architecture."""
        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)
            def forward(self, x):
                return (self.linear(x),)

        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockLayer() for _ in range(4)])
            def forward(self, input_ids, **kwargs):
                x = torch.randn(input_ids.shape[0], input_ids.shape[1], 32)
                for layer in self.layers:
                    x = layer(x)[0]
                return x

        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.randint(1, 100, (len(texts), 16)),
                    "attention_mask": torch.ones((len(texts), 16), dtype=torch.long),
                }

        model = MockTransformer()
        tokenizer = MockTokenizer()
        analyzer = LayerwiseAnalyzer(model, tokenizer, device="cpu")
        results = analyzer.analyze(["Hello world", "Agent 3 refactoring test"], batch_size=2)
        
        self.assertEqual(results["num_layers"], 4)
        self.assertEqual(len(results["cosine_similarities"]), 3)
        self.assertEqual(len(results["effective_ranks"]), 4)
        self.assertEqual(len(results["max_outlier_magnitudes"]), 4)

    # -------------------------------------------------------------------------
    # 5. Grassmann / Riemannian Federated Averaging Tests
    # -------------------------------------------------------------------------

    def test_grassmann_log_exp_inversion(self):
        """Verify Grassmann exp(log(U_0, U_1)) recovers U_1 subspace."""
        np.random.seed(42)
        # Generate two random orthonormal bases in St(8, 3)
        U0, _ = np.linalg.qr(np.random.randn(8, 3))
        U1, _ = np.linalg.qr(np.random.randn(8, 3))

        delta = grassmann_log_map(U0, U1)
        self.assertEqual(delta.shape, (8, 3))
        # Tangent property: U0^T @ Delta should be approximately 0
        self.assertTrue(np.allclose(U0.T @ delta, np.zeros((3, 3)), atol=1e-6))

        U1_reconstructed = grassmann_exp_map(U0, delta)
        # Subspace distance ||P_reconstructed - P_target||_F should be negligible
        P_orig = U1 @ U1.T
        P_rec = U1_reconstructed @ U1_reconstructed.T
        self.assertTrue(np.allclose(P_orig, P_rec, atol=1e-5))

    def test_grassmann_frechet_mean_orthogonality(self):
        """Verify Grassmann Fréchet mean produces strict orthonormal basis."""
        np.random.seed(42)
        bases = [np.linalg.qr(np.random.randn(12, 4))[0] for _ in range(5)]
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        # Spectral Fréchet Mean
        mean_spectral = grassmann_frechet_mean_spectral(bases, weights)
        self.assertEqual(mean_spectral.shape, (12, 4))
        gram_spectral = mean_spectral.T @ mean_spectral
        self.assertTrue(np.allclose(gram_spectral, np.eye(4), atol=1e-6))

        # Riemannian Karcher Mean
        mean_karcher = grassmann_karcher_mean(bases, weights, max_iter=10)
        self.assertEqual(mean_karcher.shape, (12, 4))
        gram_karcher = mean_karcher.T @ mean_karcher
        self.assertTrue(np.allclose(gram_karcher, np.eye(4), atol=1e-6))

    def test_grassmann_fedavg_aggregator_end_to_end(self):
        """Verify GrassmannFedAvgAggregator handles both SVD bases and standard weights."""
        class MockSVDLinearAttention(nn.Module):
            def __init__(self):
                super().__init__()
                # Orthogonal basis U
                U, _ = torch.linalg.qr(torch.randn(16, 4))
                self.u_proj = nn.Parameter(U)
                self.bias = nn.Parameter(torch.zeros(16))

        model = MockSVDLinearAttention()
        config = FLConfig(aggregation=AggregationStrategy.GRASSMANN_FEDAVG)
        aggregator = FL_REGISTRY.resolve("grassmann_fedavg")(config)

        # Simulate 3 client updates
        client_updates = []
        for _ in range(3):
            # Client delta for basis U
            U_client, _ = torch.linalg.qr(model.u_proj.data + 0.1 * torch.randn(16, 4))
            delta_u = (U_client - model.u_proj.data).numpy()
            delta_b = np.random.randn(16) * 0.1
            client_updates.append({"u_proj": delta_u, "bias": delta_b})

        client_weights = np.array([100, 200, 100])
        new_update = aggregator.aggregate(model, client_updates, client_weights, round_num=1)
        
        self.assertIn("u_proj", new_update)
        self.assertIn("bias", new_update)
        
        # Verify resulting aggregated basis is orthonormal
        new_global_u = model.u_proj.detach().numpy() + new_update["u_proj"]
        gram = new_global_u.T @ new_global_u
        self.assertTrue(np.allclose(gram, np.eye(4), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
