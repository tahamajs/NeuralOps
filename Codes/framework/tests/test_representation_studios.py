"""
Unit tests for Representation & Internal State Studios:
- EmbeddingSpaceExplorer
- PositionalEncodingStudio
- ResidualStreamTracer
- LayerNormDynamics
"""

import math
import os
import shutil
import sys
import tempfile
import unittest

# Ensure Codes directory is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
from framework.visualization.representation_studios import (
    EmbeddingProjectionResult, EmbeddingSpaceExplorer, LayerNormDynamics,
    LayerNormStatistics, PositionalEncodingStudio, ResidualLayerTrace,
    ResidualStreamTracer, RoPERotationState, SemanticAxisResult,
    Vector2DDynamicsTrace)


class TestRepresentationStudios(unittest.TestCase):
    """Comprehensive test suite for transformer representation & internal state studios."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.d_model = 32
        self.vocab_size = 20
        self.seq_len = 16
        self.tokens = [f"tok_{i}" for i in range(self.vocab_size)]

        np.random.seed(42)
        torch.manual_seed(42)

        # Mock embeddings
        self.embeddings = np.random.randn(self.vocab_size, self.d_model).astype(np.float32)
        # Mock hidden states across 4 layers
        self.num_layers = 4
        self.hidden_states = [
            np.random.randn(self.seq_len, self.d_model).astype(np.float32) * (1.0 + 0.3 * l)
            for l in range(self.num_layers + 1)
        ]
        self.attn_outputs = [
            np.random.randn(self.seq_len, self.d_model).astype(np.float32) * 0.5
            for _ in range(self.num_layers)
        ]
        self.ffn_outputs = [
            np.random.randn(self.seq_len, self.d_model).astype(np.float32) * 0.7
            for _ in range(self.num_layers)
        ]
        self.unembed_weight = np.random.randn(self.vocab_size, self.d_model).astype(np.float32)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # -------------------------------------------------------------
    # 1. Embedding Space Explorer Tests
    # -------------------------------------------------------------
    def test_embedding_explorer_projections(self):
        explorer = EmbeddingSpaceExplorer(output_dir=self.temp_dir)

        # PCA 3D projection
        res_pca = explorer.project_embeddings(
            self.embeddings, self.tokens, method="pca", n_components=3, n_clusters=3
        )
        self.assertIsInstance(res_pca, EmbeddingProjectionResult)
        self.assertEqual(res_pca.coordinates.shape, (self.vocab_size, 3))
        self.assertIsNotNone(res_pca.explained_variance_ratio)
        self.assertEqual(len(res_pca.cluster_labels), self.vocab_size)
        self.assertTrue(0.0 <= res_pca.anisotropy_score <= 1.0)

        # t-SNE 2D projection
        res_tsne = explorer.project_embeddings(
            self.embeddings, self.tokens, method="tsne", n_components=2
        )
        self.assertEqual(res_tsne.coordinates.shape, (self.vocab_size, 2))

        # Cosine similarity matrix
        sim_mat = explorer.compute_cosine_similarity_matrix(self.embeddings)
        self.assertEqual(sim_mat.shape, (self.vocab_size, self.vocab_size))
        # Diagonal should be approx 1.0
        np.testing.assert_allclose(np.diag(sim_mat), 1.0, atol=1e-5)

        # Serialization
        res_dict = res_pca.to_dict()
        self.assertIn("coordinates", res_dict)
        self.assertEqual(len(res_dict["tokens"]), self.vocab_size)

    def test_semantic_axes_and_linear_probe(self):
        explorer = EmbeddingSpaceExplorer(output_dir=self.temp_dir)

        # Semantic axis discovery via ICA
        axes = explorer.discover_semantic_axes(self.embeddings, self.tokens, n_axes=2, method="ica")
        self.assertEqual(len(axes), 2)
        self.assertIsInstance(axes[0], SemanticAxisResult)
        self.assertEqual(len(axes[0].top_positive_tokens), 5)
        self.assertEqual(len(axes[0].top_negative_tokens), 5)

        # Custom vector arithmetic semantic axis
        custom_axis = explorer.compute_directional_semantic_axis(
            self.embeddings,
            self.tokens,
            positive_tokens=["tok_0", "tok_1"],
            negative_tokens=["tok_2", "tok_3"],
            axis_name="Sentiment Axis",
        )
        self.assertEqual(custom_axis.name, "Sentiment Axis")
        self.assertEqual(custom_axis.axis_vector.shape, (self.d_model,))
        np.testing.assert_allclose(np.linalg.norm(custom_axis.axis_vector), 1.0, atol=1e-5)

        # Linear probe
        labels = [i % 2 for i in range(self.vocab_size)]
        probe_res = explorer.train_linear_probe(self.embeddings, labels)
        self.assertIn("accuracy", probe_res)
        self.assertIn("f1", probe_res)
        self.assertTrue(0.0 <= probe_res["accuracy"] <= 1.0)

        # Generate plot
        fig_path = explorer.generate(self.embeddings, self.tokens, save_name="emb_test.png")
        self.assertTrue(os.path.exists(fig_path))

    # -------------------------------------------------------------
    # 2. Positional Encoding Studio Tests
    # -------------------------------------------------------------
    def test_positional_encoding_studio(self):
        studio = PositionalEncodingStudio(output_dir=self.temp_dir)

        # Sinusoidal matrix
        pe = studio.generate_sinusoidal_table(max_len=32, d_model=self.d_model, base=10000.0)
        self.assertEqual(pe.shape, (32, self.d_model))
        self.assertTrue(np.all(pe >= -1.0) and np.all(pe <= 1.0))

        # RoPE frequencies
        thetas = studio.compute_rope_frequencies(d_model=self.d_model, base=10000.0)
        self.assertEqual(len(thetas), self.d_model // 2)
        self.assertTrue(np.all(thetas[:-1] > thetas[1:]))  # strictly decaying frequencies

        # 2D Rotation Matrix
        rot_mat = studio.generate_rope_rotation_matrix_2d(theta=math.pi / 4.0)
        self.assertEqual(rot_mat.shape, (2, 2))
        # Orthogonality: R R^T = I and det(R) = 1
        np.testing.assert_allclose(rot_mat @ rot_mat.T, np.eye(2), atol=1e-5)
        self.assertAlmostEqual(np.linalg.det(rot_mat), 1.0, places=5)

        # Apply RoPE to vector (norm preserving)
        vec = np.random.randn(self.d_model).astype(np.float32)
        vec_norm_orig = np.linalg.norm(vec)
        vec_rope = studio.apply_rope(vec, position=5, base=10000.0)
        vec_norm_rope = np.linalg.norm(vec_rope)
        self.assertAlmostEqual(vec_norm_orig, vec_norm_rope, places=4)

        # Granular rotation state
        state = studio.get_rope_rotation_state(
            vector_2d=[1.0, 0.0],
            position=4,
            dim_pair_idx=0,
            d_model=self.d_model,
        )
        self.assertIsInstance(state, RoPERotationState)
        self.assertAlmostEqual(state.position, 4)
        self.assertIn("rotation_matrix", state.to_dict())

        # Similarity matrix
        rope_sim = studio.compute_position_similarity_matrix(encoding_type="rope", max_len=16, d_model=self.d_model)
        self.assertEqual(rope_sim.shape, (16, 16))
        # Self-similarity along diagonal
        np.testing.assert_allclose(np.diag(rope_sim), 1.0, atol=1e-5)

        # Frequency spectrum FFT
        spec = studio.analyze_frequency_spectrum(max_len=32, d_model=self.d_model)
        self.assertEqual(len(spec["frequencies"]), self.d_model // 2)
        self.assertEqual(len(spec["wavelengths"]), self.d_model // 2)

        # Generate plot
        fig_path = studio.generate(max_len=16, d_model=self.d_model, save_name="rope_test.png")
        self.assertTrue(os.path.exists(fig_path))

    # -------------------------------------------------------------
    # 3. Residual Stream Tracer Tests
    # -------------------------------------------------------------
    def test_residual_stream_tracer(self):
        tracer = ResidualStreamTracer(output_dir=self.temp_dir)

        # Residual stream trace
        traces = tracer.trace_residual_stream(
            hidden_states=self.hidden_states,
            attn_outputs=self.attn_outputs,
            ffn_outputs=self.ffn_outputs,
            unembed_weight=self.unembed_weight,
            vocab=self.tokens,
        )
        self.assertEqual(len(traces), self.num_layers)
        self.assertIsInstance(traces[0], ResidualLayerTrace)
        self.assertTrue(0.0 <= traces[0].attention_fraction <= 1.0)
        self.assertIsNotNone(traces[0].top_logits)
        self.assertEqual(len(traces[0].top_logits), 5)

        # Direction consistency matrix
        updates = [self.attn_outputs[l] + self.ffn_outputs[l] for l in range(self.num_layers)]
        dir_mat = tracer.compute_direction_consistency_matrix(updates)
        self.assertEqual(dir_mat.shape, (self.num_layers, self.num_layers))
        np.testing.assert_allclose(np.diag(dir_mat), 1.0, atol=1e-5)

        # 2D Vector dynamics simulation
        sim_tokens = ["Alpha", "Beta", "Gamma"]
        trajectories = tracer.simulate_2d_vector_dynamics(sim_tokens, num_layers=4)
        self.assertEqual(len(trajectories), 3)
        self.assertIsInstance(trajectories[0], Vector2DDynamicsTrace)
        self.assertEqual(len(trajectories[0].layer_states), 5)  # initial + 4 layers
        self.assertEqual(len(trajectories[0].attention_deltas), 4)

        # Logit Prisms decomposition
        component_map = {
            "Embedding": self.hidden_states[0],
            "Attn_Layer0": self.attn_outputs[0],
            "FFN_Layer0": self.ffn_outputs[0],
        }
        contribs = tracer.compute_logit_prisms_decomposition(
            component_map, self.unembed_weight, target_token_idx=2
        )
        self.assertIn("Embedding", contribs)
        self.assertIn("Attn_Layer0", contribs)

        # Generate plot
        fig_path = tracer.generate(
            self.hidden_states, self.attn_outputs, self.ffn_outputs, save_name="residual_test.png"
        )
        self.assertTrue(os.path.exists(fig_path))

    # -------------------------------------------------------------
    # 4. Layer Normalization Dynamics Tests
    # -------------------------------------------------------------
    def test_layer_norm_dynamics(self):
        dynamics = LayerNormDynamics(output_dir=self.temp_dir)

        # LayerNorm computation
        vec = np.random.randn(8, self.d_model).astype(np.float32) * 5.0 + 3.0
        out_ln, inter_ln = dynamics.compute_layer_norm(vec)
        self.assertEqual(out_ln.shape, (8, self.d_model))
        # Mean should be ~ 0 and variance ~ 1
        np.testing.assert_allclose(np.mean(out_ln, axis=-1), 0.0, atol=1e-4)
        np.testing.assert_allclose(np.var(out_ln, axis=-1), 1.0, atol=1e-3)

        # RMSNorm computation
        out_rms, inter_rms = dynamics.compute_rms_norm(vec)
        self.assertEqual(out_rms.shape, (8, self.d_model))
        # Mean square should be ~ 1
        np.testing.assert_allclose(np.mean(out_rms ** 2, axis=-1), 1.0, atol=1e-3)

        # Layer statistics analysis
        pre_states = [np.random.randn(8, self.d_model).astype(np.float32) for _ in range(3)]
        post_states = [dynamics.compute_layer_norm(s)[0] for s in pre_states]
        gammas = [np.ones(self.d_model, dtype=np.float32) * 1.1 for _ in range(3)]
        betas = [np.zeros(self.d_model, dtype=np.float32) for _ in range(3)]

        stats = dynamics.analyze_layer_statistics(pre_states, post_states, gammas=gammas, betas=betas)
        self.assertEqual(len(stats), 3)
        self.assertIsInstance(stats[0], LayerNormStatistics)
        self.assertTrue(stats[0].gamma_drift_l2 > 0.0)

        # Hypersphere geometry decomposition
        sphere_data = dynamics.compute_hypersphere_projection_geometry(vec)
        self.assertIn("target_sphere_radius", sphere_data)
        self.assertEqual(len(sphere_data["pre_norm_3d"]), 8)
        self.assertEqual(len(sphere_data["post_norm_3d"]), 8)

        # Pre vs Post LN gradient stability simulation
        grad_sim = dynamics.simulate_pre_vs_post_ln_gradients(num_layers=6, num_steps=20)
        self.assertIn("post_ln_gradients", grad_sim)
        self.assertIn("pre_ln_gradients", grad_sim)
        self.assertEqual(grad_sim["post_ln_gradients"].shape, (20, 6))

        # Generate plot
        fig_path = dynamics.generate(pre_states, post_states, save_name="ln_test.png")
        self.assertTrue(os.path.exists(fig_path))


if __name__ == "__main__":
    unittest.main()
