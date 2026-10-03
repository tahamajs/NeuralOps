"""
Unit tests for Optimization & Full Pipeline Studios:
- CircuitTracer
- OutputLogitsPredictor
- LossLandscapeExplorer
- CompressionStudio
- FullPipelineStudio
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from framework.visualization.base import VISUALIZER_REGISTRY
from framework.visualization.optimization_studios import (
    CircuitTracer, CompressionStudio, FullPipelineStudio,
    LossLandscapeExplorer, OutputLogitsPredictor)


class TestOptimizationStudios(unittest.TestCase):
    """Test suite verifying all 5 optimization & full pipeline studios."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tokens = ["The", "capital", "of", "France", "is", "Paris"]
        self.seq_len = len(self.tokens)
        self.d_model = 32
        self.n_heads = 4

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # --------------------------------------------------------------------------
    # 1. CIRCUIT TRACER TESTS
    # --------------------------------------------------------------------------
    def test_circuit_tracer_qk_ov_and_classification(self):
        tracer = CircuitTracer(output_dir=self.temp_dir)
        
        # Test QK & OV decomposition
        W_Q = np.random.randn(self.d_model, 8)
        W_K = np.random.randn(self.d_model, 8)
        W_V = np.random.randn(self.d_model, 8)
        W_O = np.random.randn(8, self.d_model)

        M_QK = tracer.extract_qk_circuit(W_Q, W_K, d_head=8)
        self.assertEqual(M_QK.shape, (self.d_model, self.d_model))

        M_OV = tracer.extract_ov_circuit(W_V, W_O)
        self.assertEqual(M_OV.shape, (self.d_model, self.d_model))

        # Test head classification
        attn = np.eye(self.seq_len)
        classification = tracer.classify_head(attn, tokens=self.tokens, W_V=W_V, W_O=W_O)
        self.assertIn("primary_type", classification)
        self.assertIn("scores", classification)
        self.assertIn("induction", classification["scores"])
        self.assertIn("previous_token", classification["scores"])

    def test_circuit_tracer_attribution_graph_and_intervention(self):
        tracer = CircuitTracer(output_dir=self.temp_dir)
        attn_layers = [np.random.uniform(0.05, 0.9, (self.n_heads, self.seq_len, self.seq_len)) for _ in range(2)]

        graph = tracer.build_attribution_graph(self.tokens, attn_layers, prune_threshold=0.05)
        self.assertGreater(graph["total_nodes"], 0)
        self.assertGreater(graph["total_edges"], 0)
        self.assertEqual(len(graph["supernodes"]), 3)

        # Plot & Export
        plot_p = tracer.plot_circuit_graph(graph, save=True)
        self.assertTrue(os.path.exists(plot_p))
        json_p = tracer.export_circuit_json(graph, "test_circuit.json")
        self.assertTrue(os.path.exists(json_p))

        # Intervention simulation
        logits = np.random.randn(50)
        inter = tracer.simulate_intervention(logits, layer_idx=0, head_idx=1, ablation_type="zero")
        self.assertIn("kl_divergence", inter)
        self.assertIn("logit_diff", inter)
        self.assertGreaterEqual(inter["kl_divergence"], 0.0)

    # --------------------------------------------------------------------------
    # 2. OUTPUT LOGITS PREDICTOR TESTS
    # --------------------------------------------------------------------------
    def test_output_logits_predictor_probabilities(self):
        predictor = OutputLogitsPredictor(output_dir=self.temp_dir)
        raw_logits = np.array([2.5, 1.0, 0.5, -1.2, 3.8, 0.1])

        # Test temperature, Top-K, Top-P, repetition penalty
        res = predictor.compute_probabilities(
            raw_logits,
            temperature=0.8,
            top_k=3,
            top_p=0.9,
            repetition_penalty=1.2,
            past_token_ids=[0],
            vocab_tokens=["the", "a", "an", "of", "Paris", "in"]
        )
        self.assertEqual(len(res["top_tokens"]), min(10, len(raw_logits)))
        self.assertAlmostEqual(sum(res["probabilities"]), 1.0, places=4)
        self.assertGreaterEqual(res["entropy"], 0.0)

        # Plot
        plot_p = predictor.plot_probability_distribution(res, save=True)
        self.assertTrue(os.path.exists(plot_p))

    def test_output_logits_tuned_lens_and_calibration(self):
        predictor = OutputLogitsPredictor(output_dir=self.temp_dir)
        
        # Test Tuned / Logit Lens progression
        hidden_states = [np.random.randn(self.seq_len, self.d_model) for _ in range(4)]
        unembed = np.random.randn(20, self.d_model)
        lens_res = predictor.track_entropy_progression(hidden_states, unembed)
        self.assertEqual(lens_res["num_layers"], 4)
        self.assertEqual(len(lens_res["layer_predictions"]), 4)

        # Test Calibration metrics & Reliability diagram
        probs = np.linspace(0.1, 0.9, 50)
        labels = (probs > 0.5).astype(float)
        calib = predictor.compute_calibration_metrics(probs, labels, num_bins=5)
        self.assertIn("ece", calib)
        self.assertIn("mce", calib)
        self.assertIn("brier_score", calib)
        
        calib_plot = predictor.plot_reliability_diagram(calib, save=True)
        self.assertTrue(os.path.exists(calib_plot))

        # Beam search simulation
        beam = predictor.simulate_beam_search(raw_logits=np.random.randn(10), beam_width=2, depth=2)
        self.assertGreater(len(beam["nodes"]), 0)
        self.assertGreater(len(beam["edges"]), 0)

    # --------------------------------------------------------------------------
    # 3. LOSS LANDSCAPE EXPLORER TESTS
    # --------------------------------------------------------------------------
    def test_loss_landscape_filter_norm_and_grid(self):
        explorer = LossLandscapeExplorer(output_dir=self.temp_dir)
        weights = [np.random.randn(16, 16), np.random.randn(16)]

        # Test Filter-Normalized directions
        d1, d2 = explorer.generate_filter_normalized_directions(weights, seed=123)
        self.assertEqual(len(d1), 2)
        self.assertEqual(len(d2), 2)
        self.assertAlmostEqual(np.linalg.norm(d1[0]), np.linalg.norm(weights[0]), places=4)

        # Test 2D Loss grid sampling
        def mock_loss(w_list):
            return float(sum(np.sum(w ** 2) for w in w_list))

        grid = explorer.sample_2d_loss_grid(mock_loss, weights, d1, d2, resolution=10)
        self.assertEqual(grid["resolution"], 10)
        self.assertIn("sharpness_sam", grid)

        # Test Hessian spectrum
        hess = explorer.compute_hessian_spectrum(weights, mock_loss, num_eigenvalues=5)
        self.assertEqual(len(hess["eigenvalues"]), 5)
        self.assertGreater(hess["trace"], 0)

        # Plot 3D surface, contour, and Hessian spectrum
        surf_p = explorer.plot_3d_surface(grid, save=True)
        self.assertTrue(os.path.exists(surf_p))
        cont_p = explorer.plot_contour_map(grid, save=True)
        self.assertTrue(os.path.exists(cont_p))
        hess_p = explorer.plot_hessian_spectrum(hess, save=True)
        self.assertTrue(os.path.exists(hess_p))

    def test_loss_landscape_mode_connectivity(self):
        explorer = LossLandscapeExplorer(output_dir=self.temp_dir)
        w_a = [np.zeros((8, 8))]
        w_b = [np.ones((8, 8))]
        def loss_fn(w):
            return float(np.sum(w[0] ** 2))

        lmc = explorer.compute_mode_connectivity(w_a, w_b, loss_fn, num_steps=11, curve_type="linear")
        self.assertEqual(len(lmc["loss_profile"]), 11)
        self.assertIn("barrier_height", lmc)

    # --------------------------------------------------------------------------
    # 4. COMPRESSION STUDIO TESTS
    # --------------------------------------------------------------------------
    def test_compression_studio_engines_and_pareto(self):
        studio = CompressionStudio(output_dir=self.temp_dir)
        compressors = studio.get_available_compressors()
        self.assertGreaterEqual(len(compressors), 25)

        # Test Benchmark Simulation
        arch = {"total_parameters": 100_000_000, "base_latency_ms": 10.0}
        bench = studio.simulate_compression_benchmark(arch)
        self.assertGreater(bench["total_models"], 0)
        self.assertGreater(len(bench["pareto_frontier"]), 0)

        # Test Pareto Frontier computation
        pareto = studio.compute_pareto_frontier(bench["benchmark_results"], "compression_ratio", "accuracy_retention")
        self.assertGreater(len(pareto), 0)

        # Plot Pareto
        plot_p = studio.plot_pareto_frontier(bench, save=True)
        self.assertTrue(os.path.exists(plot_p))

    def test_compression_studio_budget_and_weight_impact(self):
        studio = CompressionStudio(output_dir=self.temp_dir)
        
        # Test Budget Allocation
        layer_sizes = [50.0, 50.0, 50.0, 50.0]
        importance = [0.9, 0.6, 0.4, 0.8]
        budget_res = studio.solve_budget_allocation(layer_sizes, importance, target_budget_mb=100.0)
        self.assertTrue(budget_res["feasible"])
        self.assertAlmostEqual(budget_res["allocated_size_mb"], 100.0, places=1)

        # Test Weight Impact
        w_orig = np.random.randn(32, 32)
        w_comp = w_orig + np.random.normal(0, 0.05, (32, 32))
        impact = studio.compute_weight_impact(w_orig, w_comp)
        self.assertIn("frobenius_error", impact)
        self.assertIn("top_affected", impact)
        self.assertGreater(impact["energy_retained"], 0.0)

    # --------------------------------------------------------------------------
    # 5. FULL PIPELINE STUDIO TESTS
    # --------------------------------------------------------------------------
    def test_full_pipeline_studio_forward_trace(self):
        pipeline = FullPipelineStudio(output_dir=self.temp_dir)
        trace = pipeline.trace_forward_pass(self.tokens, d_model=32, n_heads=4, n_layers=2, vocab_size=50)

        self.assertEqual(len(trace["layers"]), 2)
        self.assertEqual(len(trace["predictions"]), 5)
        self.assertIn("profiling", trace)
        self.assertIn("stage_latencies_ms", trace["profiling"])

        # Drift matrix
        drift = pipeline.compute_representation_drift(trace)
        self.assertEqual(drift.shape, (3, 3))

        # Gamified Challenge reconstruction
        user_recon = {"layers": [{"hidden_out_norm": trace["layers"][0]["hidden_out_norm"] * 0.98}]}
        eval_res = pipeline.evaluate_challenge_reconstruction(trace, user_recon)
        self.assertGreaterEqual(eval_res["reconstruction_score"], 0.0)
        self.assertIn("rank_badge", eval_res)

        # Plots and Export
        flow_p = pipeline.plot_pipeline_flow(trace, save=True)
        self.assertTrue(os.path.exists(flow_p))
        lat_p = pipeline.plot_latency_breakdown(trace["profiling"], save=True)
        self.assertTrue(os.path.exists(lat_p))
        json_p = pipeline.export_pipeline_trace_json(trace)
        self.assertTrue(os.path.exists(json_p))

    # --------------------------------------------------------------------------
    # 6. VISUALIZER REGISTRY INTEGRATION TEST
    # --------------------------------------------------------------------------
    def test_visualizer_registry_membership(self):
        self.assertIn("circuit_tracer", VISUALIZER_REGISTRY)
        self.assertIn("output_logits_predictor", VISUALIZER_REGISTRY)
        self.assertIn("loss_landscape_explorer", VISUALIZER_REGISTRY)
        self.assertIn("compression_studio", VISUALIZER_REGISTRY)
        self.assertIn("full_pipeline_studio", VISUALIZER_REGISTRY)


if __name__ == "__main__":
    unittest.main()
