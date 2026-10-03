"""
Unit tests for the Attention Analytics & Interpretability Suite.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from framework.visualization.interpretability_suite import (
    Head3DMetricsResult, HeadAblationCalculator, InterpretabilitySuite,
    calculate_head_importance, calculate_head_similarity,
    calculate_head_stability, classify_head_type, decompose_circuit_matrix,
    generate_attribution_graph, generate_head_3d_metrics,
    generate_logit_lens_timeline, generate_semantic_pathways)


class TestInterpretabilitySuite(unittest.TestCase):
    """Comprehensive test suite for Zone 5 & Zone 6 interpretability algorithms."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.num_layers = 4
        self.num_heads = 4
        self.seq_len = 8
        self.d_model = 32
        self.vocab_size = 50

        np.random.seed(42)
        torch.manual_seed(42)

        # Mock attention matrices (num_layers, num_heads, seq_len, seq_len)
        raw_attn = np.random.uniform(0.01, 1.0, size=(self.num_layers, self.num_heads, self.seq_len, self.seq_len))
        # Softmax normalize over key dimension
        self.mock_attn = raw_attn / np.sum(raw_attn, axis=-1, keepdims=True)

        self.mock_tokens = ["The", "neural", "network", "computes", "attention", "over", "tokens", "."]
        self.mock_vocab = [f"word_{i}" for i in range(self.vocab_size)]

        # Mock hidden states across layers (num_layers, seq_len, d_model)
        self.mock_hidden = np.random.randn(self.num_layers, self.seq_len, self.d_model).astype(np.float32)
        # Mock unembedding matrix (vocab_size, d_model)
        self.mock_unembed = np.random.randn(self.vocab_size, self.d_model).astype(np.float32)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_classify_head_type(self):
        # 1. Positional diagonal matrix
        diag_mat = np.eye(self.seq_len)
        htype, color, desc = classify_head_type(diag_mat, tokens=self.mock_tokens)
        self.assertIn("Positional", htype)
        self.assertTrue(color.startswith("#"))

        # 2. Delimiter sink matrix
        sink_mat = np.zeros((self.seq_len, self.seq_len))
        sink_mat[:, 0] = 1.0
        htype, color, _ = classify_head_type(sink_mat, tokens=self.mock_tokens)
        self.assertIn("Delimiter", htype)

        # 3. Uniform diffuse matrix
        uniform_mat = np.ones((self.seq_len, self.seq_len)) / self.seq_len
        htype, color, _ = classify_head_type(uniform_mat, tokens=self.mock_tokens)
        self.assertTrue("Redundant" in htype or "Mixed" in htype or "Diffuse" in htype)

    def test_head_importance_and_stability(self):
        imp = calculate_head_importance(self.mock_attn)
        self.assertEqual(imp.shape, (self.num_layers, self.num_heads))
        self.assertTrue(np.all(imp >= 0.0) and np.all(imp <= 1.0))

        # Stability across 3 calibration inputs
        calib = [self.mock_attn, self.mock_attn * 0.9 + 0.1 / self.seq_len, self.mock_attn]
        stab = calculate_head_stability(calib)
        self.assertEqual(stab.shape, (self.num_layers, self.num_heads))
        self.assertTrue(np.all(stab >= 0.0) and np.all(stab <= 1.0))

        sim = calculate_head_similarity(self.mock_attn)
        self.assertEqual(sim.shape, (self.num_layers, self.num_heads))
        self.assertTrue(np.all(sim >= 0.0) and np.all(sim <= 1.0))

    def test_generate_head_3d_metrics(self):
        res: Head3DMetricsResult = generate_head_3d_metrics(
            self.mock_attn,
            tokens=self.mock_tokens,
        )
        self.assertEqual(len(res.points), self.num_layers * self.num_heads)
        self.assertEqual(res.layers, self.num_layers)
        self.assertEqual(res.heads_per_layer, self.num_heads)
        self.assertGreater(len(res.type_distribution), 0)

        # Check serialization
        d = res.to_dict()
        self.assertIn("points", d)
        self.assertIn("avg_importance", d)
        self.assertEqual(len(d["points"]), self.num_layers * self.num_heads)

    def test_head_ablation_calculator(self):
        calculator = HeadAblationCalculator(model_id="test-bert")

        clean_logits = np.random.randn(self.seq_len, self.vocab_size)
        ablated_logits = clean_logits + np.random.normal(0, 0.2, size=clean_logits.shape)
        # Force a prediction change at pos 0
        ablated_logits[0, :] = 0.0
        ablated_logits[0, 5] = 10.0

        res = calculator.calculate_ablation_impact(
            clean_logits=clean_logits,
            ablated_logits=ablated_logits,
            ablated_heads=[(1, 2), (2, 3)],
            tokens=self.mock_tokens,
        )

        self.assertGreaterEqual(res.kl_divergence, 0.0)
        self.assertGreaterEqual(res.top_k_flips, 1)
        self.assertGreaterEqual(len(res.affected_tokens), 1)

        # Test pruning recipe generation
        head_metrics = generate_head_3d_metrics(self.mock_attn, tokens=self.mock_tokens)
        recipe = calculator.generate_pruning_recipe(head_metrics, threshold_importance=0.5)
        self.assertEqual(recipe.model_id, "test-bert")
        self.assertIsInstance(recipe.ablated_heads, list)
        self.assertIn("pruning_ratio", recipe.to_dict())

    def test_circuit_decomposition_and_attribution(self):
        W_QK = np.random.randn(self.d_model, self.d_model)
        svd_res = decompose_circuit_matrix(W_QK, layer=0, head=1, circuit_type="QK")

        self.assertEqual(svd_res.layer, 0)
        self.assertEqual(svd_res.head, 1)
        self.assertEqual(svd_res.circuit_type, "QK")
        self.assertGreater(len(svd_res.singular_values), 0)
        self.assertGreater(svd_res.effective_rank, 0.0)
        self.assertGreater(svd_res.spectral_radius, 0.0)

        # Attribution Graph
        metrics = generate_head_3d_metrics(self.mock_attn, tokens=self.mock_tokens)
        graph = generate_attribution_graph(metrics, threshold_weight=0.1)
        self.assertGreater(len(graph.nodes), 0)
        self.assertIsInstance(graph.edges, list)

    def test_semantic_pathways(self):
        pathway_res = generate_semantic_pathways(
            hidden_states=self.mock_hidden,
            tokens=self.mock_tokens,
            attention_weights=self.mock_attn,
        )

        self.assertEqual(len(pathway_res.tokens), len(self.mock_tokens))
        self.assertEqual(len(pathway_res.trajectories), len(self.mock_tokens))
        # Each trajectory has 2D coordinates for all layers
        for traj in pathway_res.trajectories:
            self.assertEqual(len(traj.coordinates), self.num_layers)
            self.assertEqual(len(traj.coordinates[0]), 2)
            self.assertEqual(len(traj.velocities), self.num_layers - 1)
            self.assertGreaterEqual(traj.total_path_length, 0.0)

        d = pathway_res.to_dict()
        self.assertIn("trajectories", d)
        self.assertIn("explained_variance_ratio", d)

    def test_logit_lens_timeline(self):
        ll_res = generate_logit_lens_timeline(
            hidden_states=self.mock_hidden,
            unembedding_matrix=self.mock_unembed,
            vocabulary=self.mock_vocab,
            token_position=3,
            top_k=5,
        )

        self.assertEqual(ll_res.token_position, 3)
        self.assertEqual(len(ll_res.layer_predictions), self.num_layers)
        for lp in ll_res.layer_predictions:
            self.assertEqual(len(lp.top_k_tokens), 5)
            self.assertEqual(len(lp.top_k_probs), 5)
            self.assertGreaterEqual(lp.entropy, 0.0)

        self.assertGreaterEqual(ll_res.convergence_layer, 0)
        self.assertLess(ll_res.convergence_layer, self.num_layers)

    def test_full_interpretability_suite(self):
        suite = InterpretabilitySuite(output_dir=self.temp_dir)
        payload = suite.run_suite(
            attention_matrices=self.mock_attn,
            hidden_states=self.mock_hidden,
            unembedding_matrix=self.mock_unembed,
            tokens=self.mock_tokens,
            vocabulary=self.mock_vocab,
        )

        self.assertIn("head_metrics", payload)
        self.assertIn("pruning_recipe", payload)
        self.assertIn("attribution_graph", payload)
        self.assertIn("semantic_pathways", payload)
        self.assertIn("logit_lens", payload)

        json_path = suite.export_summary_json(payload, "test_summary.json")
        self.assertTrue(os.path.exists(json_path))


if __name__ == "__main__":
    unittest.main()
