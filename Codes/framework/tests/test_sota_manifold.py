"""
Comprehensive unit tests for SOTA Manifold Optimization and Drift Detection:
1. Tangent space Riemannian gradient projection and QR retraction on Stiefel manifold.
2. Orthogonality preservation ||W^T W - I|| < 1e-5 over multi-step Riemannian optimization.
3. Automatic detection of orthogonal low-rank matrices (U, V) and hybrid AdamW/SGD Euclidean updates.
4. Hutchinson randomized trace estimator with Rademacher and Gaussian distributions.
5. Nyström low-rank matrix square root and fast 2-Wasserstein (Bures-Wasserstein) metric calculations.
6. DriftDetector integration with registry methods.
"""

from __future__ import annotations

import math
import unittest
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from framework.drift.config import DriftConfig
from framework.drift.detectors import (METHOD_REGISTRY, DriftDetector,
                                       bures_wasserstein_distance,
                                       fast_bures_wasserstein_nystrom,
                                       hutchinson_trace_estimator,
                                       matrix_square_root,
                                       nystrom_low_rank_approximation,
                                       nystrom_matrix_square_root)
from framework.trainer.optimizers import (OPTIMIZER_REGISTRY,
                                          RiemannianStiefelOptimizer,
                                          is_stiefel_parameter,
                                          project_stiefel_gradient,
                                          stiefel_qr_retraction)


class TestStiefelManifoldGeometry(unittest.TestCase):
    """Unit tests for Stiefel manifold differential geometry operations."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    def test_project_stiefel_gradient_tangent_orthogonality(self) -> None:
        """
        Verify that tangent space projection produces a tangent vector satisfying
        skew-symmetry: sym(W^T @ grad_R) = (W^T @ grad_R + grad_R^T @ W) / 2 = 0.
        """
        # Tall matrix n=16, p=4
        n, p = 16, 4
        x = torch.randn(n, p, dtype=torch.float64)
        q, _ = torch.linalg.qr(x)
        w = q[:, :p]  # W in St(p, n)

        # Arbitrary Euclidean gradient G
        g = torch.randn(n, p, dtype=torch.float64)

        # Project to tangent space
        # RESEARCH FIX: Validate skew-symmetry of projected Riemannian gradient
        grad_r = project_stiefel_gradient(w, g)

        # Check sym(W^T @ grad_R) = 0
        wt_grad = torch.matmul(w.t(), grad_r)
        sym_check = 0.5 * (wt_grad + wt_grad.t())
        sym_norm = torch.linalg.norm(sym_check).item()
        self.assertLess(sym_norm, 1e-10, f"Tangent projection skew-symmetry violated: {sym_norm}")

    def test_project_stiefel_gradient_wide_matrix(self) -> None:
        """Verify tangent space projection for row-orthonormal matrices W in R^{n x p} with n < p."""
        n, p = 4, 16
        x = torch.randn(p, n, dtype=torch.float64)
        q, _ = torch.linalg.qr(x)
        w = q[:, :n].t()  # W in R^{4 x 16}, W @ W^T = I_4

        g = torch.randn(n, p, dtype=torch.float64)
        grad_r = project_stiefel_gradient(w, g)

        # Check sym(grad_R @ W^T) = 0
        grad_wt = torch.matmul(grad_r, w.t())
        sym_check = 0.5 * (grad_wt + grad_wt.t())
        sym_norm = torch.linalg.norm(sym_check).item()
        self.assertLess(sym_norm, 1e-10, f"Wide matrix tangent projection violated: {sym_norm}")

    def test_project_stiefel_gradient_batched(self) -> None:
        """Verify batched tangent space projection across batch dimensions."""
        b, n, p = 3, 16, 4
        x = torch.randn(b, n, p, dtype=torch.float64)
        q, _ = torch.linalg.qr(x)
        w = q[:, :, :p]
        g = torch.randn(b, n, p, dtype=torch.float64)

        grad_r = project_stiefel_gradient(w, g)
        wt_grad = torch.matmul(w.transpose(-2, -1), grad_r)
        sym_check = 0.5 * (wt_grad + wt_grad.transpose(-2, -1))
        sym_norm = torch.linalg.norm(sym_check).item()
        self.assertLess(sym_norm, 1e-10, f"Batched tangent projection violated: {sym_norm}")

    def test_stiefel_qr_retraction_orthogonality(self) -> None:
        """
        Verify that QR retraction strictly restores exact orthogonality
        ||W_next^T @ W_next - I_p||_F < 1e-12.
        """
        n, p = 20, 6
        x = torch.randn(n, p, dtype=torch.float64)
        q, _ = torch.linalg.qr(x)
        w = q[:, :p]

        g = torch.randn(n, p, dtype=torch.float64)
        grad_r = project_stiefel_gradient(w, g)

        # Step size lr = 0.5
        # RESEARCH FIX: Validate exact machine-precision orthogonality retention under QR retraction
        w_next = stiefel_qr_retraction(w, grad_r, lr=0.5)

        gram = torch.matmul(w_next.t(), w_next)
        eye = torch.eye(p, dtype=torch.float64)
        ortho_err = torch.linalg.norm(gram - eye).item()
        self.assertLess(ortho_err, 1e-12, f"QR retraction orthogonality error: {ortho_err}")

    def test_stiefel_qr_retraction_wide_orthogonality(self) -> None:
        """Verify QR retraction for row-orthonormal wide matrices."""
        n, p = 6, 20
        x = torch.randn(p, n, dtype=torch.float64)
        q, _ = torch.linalg.qr(x)
        w = q[:, :n].t()

        g = torch.randn(n, p, dtype=torch.float64)
        grad_r = project_stiefel_gradient(w, g)

        w_next = stiefel_qr_retraction(w, grad_r, lr=0.5)
        gram = torch.matmul(w_next, w_next.t())
        eye = torch.eye(n, dtype=torch.float64)
        ortho_err = torch.linalg.norm(gram - eye).item()
        self.assertLess(ortho_err, 1e-12, f"Wide QR retraction error: {ortho_err}")


class ToySVDLayer(nn.Module):
    """Toy factorized SVD layer with orthogonal U, V parameters and Euclidean bias."""

    def __init__(self, in_features: int = 32, out_features: int = 16, rank: int = 4) -> None:
        super().__init__()
        # Initialize orthonormal U in St(rank, out_features)
        u_init = torch.randn(out_features, rank)
        q_u, _ = torch.linalg.qr(u_init)
        self.u_proj = nn.Parameter(q_u[:, :rank].clone())

        # Initialize orthonormal V in St(rank, in_features)
        v_init = torch.randn(in_features, rank)
        q_v, _ = torch.linalg.qr(v_init)
        self.v_proj = nn.Parameter(q_v[:, :rank].clone())

        # Singular values / diagonal scaling
        self.s_diag = nn.Parameter(torch.ones(rank))

        # Euclidean bias and output head
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.classifier = nn.Linear(out_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Factorized forward: x @ V @ diag(S) @ U^T + bias
        v_scaled = self.v_proj * self.s_diag.unsqueeze(0)  # (in_features, rank)
        h = torch.matmul(x, v_scaled)  # (batch, rank)
        h = torch.matmul(h, self.u_proj.t()) + self.bias  # (batch, out_features)
        h = self.norm(h)
        return self.classifier(h)


class TestRiemannianStiefelOptimizer(unittest.TestCase):
    """Unit tests for RiemannianStiefelOptimizer multi-step training and automatic parameter partitioning."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    def test_auto_parameter_partitioning(self) -> None:
        """Verify that U, V parameters are automatically assigned to Stiefel group and others to Euclidean."""
        model = ToySVDLayer(in_features=32, out_features=16, rank=4)
        opt = RiemannianStiefelOptimizer(model, lr=1e-3, lr_stiefel=1e-2, base_optimizer="adamw")

        self.assertEqual(len(opt.param_groups), 2)
        stiefel_group = next(g for g in opt.param_groups if g.get("manifold") == "stiefel")
        euclidean_group = next(g for g in opt.param_groups if g.get("manifold") == "euclidean")

        self.assertEqual(stiefel_group["lr"], 1e-2)
        self.assertEqual(euclidean_group["lr"], 1e-3)
        self.assertEqual(stiefel_group["weight_decay"], 0.0)

        stiefel_params = set(stiefel_group["params"])
        self.assertIn(model.u_proj, stiefel_params)
        self.assertIn(model.v_proj, stiefel_params)

        euclidean_params = set(euclidean_group["params"])
        self.assertIn(model.bias, euclidean_params)
        self.assertIn(model.s_diag, euclidean_params)
        self.assertIn(model.classifier.weight, euclidean_params)

    def test_orthogonality_preservation_multistep(self) -> None:
        """
        Verify that ||W^T W - I|| < 1e-5 is strictly preserved after 100 optimization steps
        with aggressive learning rates and stochastic gradients.
        """
        model = ToySVDLayer(in_features=32, out_features=16, rank=4)
        optimizer = RiemannianStiefelOptimizer(
            model, lr=5e-2, lr_stiefel=5e-2, base_optimizer="adamw"
        )

        for step in range(100):
            optimizer.zero_grad()
            batch_x = torch.randn(8, 32)
            targets = torch.randint(0, 2, (8,))
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        # RESEARCH FIX: Assert strict orthogonality preservation ||W^T W - I|| < 1e-5
        u_err = optimizer.check_orthogonality_error(model.u_proj)
        v_err = optimizer.check_orthogonality_error(model.v_proj)

        self.assertLess(u_err, 1e-5, f"U orthogonality error exceeded 1e-5: {u_err}")
        self.assertLess(v_err, 1e-5, f"V orthogonality error exceeded 1e-5: {v_err}")

        all_errors = optimizer.check_orthogonality_error()
        for idx, err in all_errors.items():
            self.assertLess(err, 1e-5, f"Manifold parameter {idx} orthogonality error: {err}")

    def test_stiefel_with_riemannian_momentum(self) -> None:
        """Verify Riemannian optimization with non-zero momentum maintains exact orthogonality."""
        model = ToySVDLayer(in_features=32, out_features=16, rank=4)
        optimizer = RiemannianStiefelOptimizer(
            model, lr=2e-2, lr_stiefel=2e-2, stiefel_momentum=0.9, base_optimizer="sgd"
        )

        for _ in range(50):
            optimizer.zero_grad()
            batch_x = torch.randn(8, 32)
            targets = torch.randint(0, 2, (8,))
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        u_err = optimizer.check_orthogonality_error(model.u_proj)
        v_err = optimizer.check_orthogonality_error(model.v_proj)
        self.assertLess(u_err, 1e-5, f"U momentum error: {u_err}")
        self.assertLess(v_err, 1e-5, f"V momentum error: {v_err}")

    def test_hybrid_sgd_euclidean_optimizer(self) -> None:
        """Verify RiemannianStiefelOptimizer works correctly with base_optimizer='sgd'."""
        model = ToySVDLayer(in_features=16, out_features=8, rank=2)
        optimizer = RiemannianStiefelOptimizer(
            model, lr=1e-2, base_optimizer="sgd", momentum=0.9, weight_decay=0.01
        )

        batch_x = torch.randn(4, 16)
        targets = torch.randint(0, 2, (4,))
        loss = nn.functional.cross_entropy(model(batch_x), targets)
        loss.backward()
        optimizer.step()

        # Check state initialized for Euclidean parameter
        state = optimizer.state[model.bias]
        self.assertIn("momentum_buffer", state)
        self.assertEqual(state["step"], 1)

    def test_optimizer_registry_integration(self) -> None:
        """Verify RiemannianStiefelOptimizer can be instantiated via OPTIMIZER_REGISTRY."""
        model = ToySVDLayer(in_features=16, out_features=8, rank=2)
        factory = OPTIMIZER_REGISTRY.get("riemannian_stiefel")
        self.assertIsNotNone(factory)
        opt = factory(model, lr=1e-3)
        self.assertIsInstance(opt, RiemannianStiefelOptimizer)


class TestHutchinsonTraceEstimator(unittest.TestCase):
    """Unit tests for Hutchinson's randomized trace estimator."""

    def setUp(self) -> None:
        np.random.seed(42)

    def test_hutchinson_trace_rademacher(self) -> None:
        """Verify Hutchinson trace estimation accuracy with Rademacher probe vectors."""
        d = 100
        # Generate symmetric positive definite matrix with known trace
        q, _ = np.linalg.qr(np.random.randn(d, d))
        eigenvalues = np.linspace(0.1, 10.0, d)
        a = q @ np.diag(eigenvalues) @ q.T
        exact_trace = float(np.sum(eigenvalues))

        # RESEARCH FIX: Evaluate Hutchinson randomized trace estimator convergence
        est_trace = hutchinson_trace_estimator(
            a, num_samples=200, distribution="rademacher", seed=42
        )
        rel_error = abs(est_trace - exact_trace) / exact_trace
        self.assertLess(rel_error, 0.08, f"Rademacher relative error too high: {rel_error}")

    def test_hutchinson_trace_gaussian(self) -> None:
        """Verify Hutchinson trace estimation accuracy with Gaussian probe vectors."""
        d = 100
        q, _ = np.linalg.qr(np.random.randn(d, d))
        eigenvalues = np.linspace(1.0, 20.0, d)
        a = q @ np.diag(eigenvalues) @ q.T
        exact_trace = float(np.sum(eigenvalues))

        est_trace = hutchinson_trace_estimator(
            a, num_samples=200, distribution="gaussian", seed=42
        )
        rel_error = abs(est_trace - exact_trace) / exact_trace
        self.assertLess(rel_error, 0.08, f"Gaussian relative error too high: {rel_error}")

    def test_hutchinson_callable_function(self) -> None:
        """Verify Hutchinson estimator works with implicit matrix-vector product callables."""
        d = 50
        a = np.diag(np.arange(1, d + 1, dtype=np.float64))
        exact_trace = float(np.sum(np.arange(1, d + 1)))

        def matvec(v: np.ndarray) -> np.ndarray:
            return a @ v

        est_trace = hutchinson_trace_estimator(
            matvec, dim=d, num_samples=150, distribution="rademacher", seed=42
        )
        rel_error = abs(est_trace - exact_trace) / exact_trace
        self.assertLess(rel_error, 0.08, f"Callable matvec relative error too high: {rel_error}")


class TestNystromWassersteinApproximation(unittest.TestCase):
    """Unit tests for Nyström low-rank approximation and 2-Wasserstein matrix square root calculation."""

    def setUp(self) -> None:
        np.random.seed(42)

    def test_nystrom_low_rank_reconstruction(self) -> None:
        """Verify Nyström low-rank decomposition accurately captures top spectrum."""
        d, r_true = 80, 5
        # Low-rank ground truth + small noise
        u_true, _ = np.linalg.qr(np.random.randn(d, r_true))
        s_true = np.array([50.0, 30.0, 20.0, 10.0, 5.0])
        cov = u_true @ np.diag(s_true) @ u_true.T + 1e-4 * np.eye(d)

        # RESEARCH FIX: Validate Nyström low-rank approximation of PSD covariance
        u_r, lambda_r = nystrom_low_rank_approximation(cov, rank=5, seed=42)
        self.assertEqual(u_r.shape, (d, 5))
        self.assertEqual(lambda_r.shape, (5,))

        # Orthonormality of U_r
        gram = u_r.T @ u_r
        eye = np.eye(5)
        self.assertLess(np.linalg.norm(gram - eye), 1e-10)

        # Captured trace should match top-5 singular values closely
        captured_trace = np.sum(lambda_r)
        true_top_trace = np.sum(s_true)
        self.assertAlmostEqual(captured_trace, true_top_trace, delta=0.5)

    def test_nystrom_matrix_square_root(self) -> None:
        """Verify Nyström matrix square root satisfies (Sigma^{1/2})^2 approx Sigma."""
        d, r = 60, 4
        u, _ = np.linalg.qr(np.random.randn(d, r))
        s = np.array([40.0, 25.0, 15.0, 5.0])
        cov = u @ np.diag(s) @ u.T

        sqrt_cov = nystrom_matrix_square_root(cov, rank=r, seed=42)
        recon = sqrt_cov @ sqrt_cov
        rel_diff = np.linalg.norm(recon - cov) / np.linalg.norm(cov)
        self.assertLess(rel_diff, 0.05, f"Matrix square root relative error: {rel_diff}")

    def test_fast_wasserstein_accuracy_vs_exact(self) -> None:
        """
        Verify that fast Nyström 2-Wasserstein calculation closely matches
        the exact O(d^3) Bures-Wasserstein distance on low-rank Gaussian distributions.
        """
        d, r = 80, 8
        u1, _ = np.linalg.qr(np.random.randn(d, r))
        s1 = np.sort(np.random.uniform(5.0, 50.0, r))[::-1]
        cov1 = u1 @ np.diag(s1) @ u1.T + 1e-4 * np.eye(d)

        u2, _ = np.linalg.qr(np.random.randn(d, r))
        s2 = np.sort(np.random.uniform(5.0, 50.0, r))[::-1]
        cov2 = u2 @ np.diag(s2) @ u2.T + 1e-4 * np.eye(d)

        mean1 = np.random.randn(d)
        mean2 = np.random.randn(d)

        # Exact O(d^3) 2-Wasserstein
        exact_w2 = bures_wasserstein_distance(cov1, cov2, mean1=mean1, mean2=mean2)

        # Fast O(d r^2 + r^3) Nyström 2-Wasserstein
        # RESEARCH FIX: Compare fast Nystrom Wasserstein to exact O(d^3) distance
        fast_w2 = fast_bures_wasserstein_nystrom(
            cov1, cov2, mean1=mean1, mean2=mean2, rank=r, seed=42
        )

        rel_diff = abs(fast_w2 - exact_w2) / exact_w2
        self.assertLess(rel_diff, 0.05, f"Fast Wasserstein relative difference too large: {rel_diff}")

    def test_drift_detector_with_fast_wasserstein(self) -> None:
        """Verify DriftDetector integration with fast Nyström 2-Wasserstein and Hutchinson trace."""
        n_samples, n_feats = 100, 6
        ref_data = {f"feat_{i}": np.random.randn(n_samples) for i in range(n_feats)}
        # Shifted current data
        cur_data = {f"feat_{i}": np.random.randn(n_samples) + 0.8 for i in range(n_feats)}

        cfg = DriftConfig(
            methods=["ks", "wasserstein_2", "wasserstein_2_nystrom", "hutchinson_trace"],
            enable_multivariate=True,
            multivariate_method="fast_wasserstein",
            threshold=0.1,
        )

        detector = DriftDetector(ref_data, config=cfg)
        report = detector.detect(cur_data)

        self.assertIsNotNone(report)
        self.assertEqual(len(report.features), n_feats)
        for feat_rep in report.features:
            self.assertIn("wasserstein_2_nystrom", feat_rep.methods)
            self.assertIn("hutchinson_trace", feat_rep.methods)
            self.assertIn("wasserstein_2", feat_rep.methods)

        self.assertIsNotNone(report.multivariate)
        self.assertEqual(report.multivariate.get("method"), "fast_wasserstein")
        self.assertTrue(report.multivariate.get("drifted"))


if __name__ == "__main__":
    unittest.main()
