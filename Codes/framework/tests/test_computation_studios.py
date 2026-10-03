"""
Unit tests for Computation, Weight & Memory Studios.
Tests FFNNeuronExplorer, KVCacheMonitor, WeightMatrixDecomposer, and LoRAAdapterStudio.
"""

import os
import shutil
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from framework.visualization.base import VISUALIZER_REGISTRY
from framework.visualization.computation_studios import (
    FFNNeuronExplorer, KVCacheMonitor, LoRAAdapterStudio,
    WeightMatrixDecomposer)


class TestComputationStudios(unittest.TestCase):
    """Comprehensive test suite for transformer computation & memory visualization studios."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        torch.manual_seed(42)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # -------------------------------------------------------------------------
    # 1. FFNNeuronExplorer Tests
    # -------------------------------------------------------------------------
    def test_ffn_activation_heatmap(self):
        explorer = FFNNeuronExplorer(output_dir=self.temp_dir)
        activations = np.random.randn(8, 64).astype(np.float32)
        tokens = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]

        res = explorer.generate_activation_heatmap(
            activations=activations,
            tokens=tokens,
            layer_idx=2,
            neuron_indices=[0, 1, 5, 10],
            save=True,
        )
        self.assertIn("matrix", res)
        self.assertEqual(res["shape"], [8, 4])
        self.assertEqual(res["tokens"], tokens)
        self.assertTrue(os.path.exists(res["file_path"]))

    def test_ffn_dead_neuron_detection(self):
        explorer = FFNNeuronExplorer(output_dir=self.temp_dir)
        # Create activations with 10 dead neurons out of 50
        activations = np.random.randn(16, 50).astype(np.float32)
        activations[:, :10] = 0.0  # dead neurons

        dead_info = explorer.detect_dead_neurons(activations, threshold=1e-6)
        self.assertEqual(dead_info["dead_count"], 10)
        self.assertEqual(dead_info["total_neurons"], 50)
        self.assertAlmostEqual(dead_info["dead_ratio"], 0.20)
        self.assertEqual(dead_info["dead_indices"], list(range(10)))
        self.assertIn(dead_info["health"], ["warning", "critical"])

    def test_ffn_top_activating_examples(self):
        explorer = FFNNeuronExplorer(output_dir=self.temp_dir)
        seq_len = 10
        activations = np.zeros((seq_len, 8), dtype=np.float32)
        activations[3, 0] = 9.5
        activations[7, 0] = 4.2
        tokens = [f"tok_{i}" for i in range(seq_len)]

        top_res = explorer.find_top_activating_examples(activations, tokens, top_k=2, neuron_indices=[0])
        self.assertIn(0, top_res)
        n0 = top_res[0]
        self.assertEqual(n0["top_examples"][0]["token_index"], 3)
        self.assertEqual(n0["top_examples"][0]["token"], "tok_3")
        self.assertAlmostEqual(n0["top_examples"][0]["activation"], 9.5, places=4)
        self.assertGreater(n0["selectivity"], 0.0)

    def test_ffn_sae_feature_breakdown(self):
        explorer = FFNNeuronExplorer(output_dir=self.temp_dir)
        X = np.random.randn(12, 32).astype(np.float32)
        W_enc = np.random.randn(32, 128).astype(np.float32)
        W_dec = np.random.randn(128, 32).astype(np.float32)

        sae_res = explorer.sae_feature_breakdown(
            activations=X,
            sae_encoder_weight=W_enc,
            sae_decoder_weight=W_dec,
            top_features=5,
        )
        self.assertEqual(sae_res["num_features"], 128)
        self.assertIn("mean_l0_sparsity", sae_res)
        self.assertEqual(len(sae_res["top_features"]), 5)
        self.assertIsNotNone(sae_res["reconstruction_error"])

    def test_ffn_statistics_and_dead_plot(self):
        explorer = FFNNeuronExplorer(output_dir=self.temp_dir)
        acts = np.random.randn(20, 16).astype(np.float32)
        stats = explorer.compute_neuron_statistics(acts)
        self.assertEqual(len(stats["mean_per_neuron"]), 16)
        self.assertEqual(len(stats["skewness_per_neuron"]), 16)

        layer_dict = {0: acts, 1: acts * 0.1, 2: acts}
        plot_path = explorer.plot_dead_neuron_distribution(layer_dict, save=True)
        self.assertIsNotNone(plot_path)
        self.assertTrue(os.path.exists(plot_path))

    # -------------------------------------------------------------------------
    # 2. KVCacheMonitor Tests
    # -------------------------------------------------------------------------
    def test_kv_cache_scaling_simulation(self):
        monitor = KVCacheMonitor(output_dir=self.temp_dir)
        seq_lens = [512, 1024, 2048, 4096]
        res = monitor.simulate_cache_growth(
            seq_lens=seq_lens,
            num_layers=12,
            num_heads=12,
            head_dim=64,
            bytes_per_elem=2.0,
            compression_methods={"DeltaKV": 0.25, "StreamingLLM": 0.20},
        )
        self.assertEqual(len(res["baseline_mb"]), 4)
        self.assertIn("DeltaKV", res["compression_curves_mb"])
        self.assertEqual(res["memory_savings_percent"]["DeltaKV"], "75.0%")

    def test_kv_cache_delta_heatmap(self):
        monitor = KVCacheMonitor(output_dir=self.temp_dir)
        k1 = np.random.randn(32, 64).astype(np.float32)
        k2 = k1 + np.random.randn(32, 64).astype(np.float32) * 0.1
        # Make first 2 tokens static (attention sink candidates)
        k2[:2] = k1[:2]

        delta_res = monitor.compute_delta_kv_heatmap(k1, k2, layer_idx=1, head_idx=0, save=True)
        self.assertIn("mean_delta", delta_res)
        self.assertIn(0, delta_res["static_token_indices"])
        self.assertTrue(os.path.exists(delta_res["file_path"]))

    def test_kv_cache_attention_sinks(self):
        monitor = KVCacheMonitor(output_dir=self.temp_dir)
        # Create attention weights where token 0 absorbs 60% mass
        attn = np.ones((8, 64, 64), dtype=np.float32)
        attn[:, :, 0] = 50.0

        sink_res = monitor.detect_attention_sinks(attn, threshold_multiplier=1.5, top_k_sinks=3)
        self.assertIn(0, sink_res["sink_indices"])
        self.assertEqual(sink_res["top_sink_candidates"][0], 0)
        self.assertGreater(sink_res["sink_mass_percentage"], 20.0)

    def test_kv_cache_compression_tracker(self):
        monitor = KVCacheMonitor(output_dir=self.temp_dir)
        orig = np.random.randn(1024, 64).astype(np.float32)
        comp = orig + 0.01 * np.random.randn(1024, 64).astype(np.float32)

        ratio_info = monitor.track_compression_ratios(orig, comp, bits_original=16, bits_compressed=4)
        self.assertAlmostEqual(ratio_info["compression_ratio"], 0.25, places=3)
        self.assertAlmostEqual(ratio_info["memory_saved_percent"], 75.0, places=1)
        self.assertIsNotNone(ratio_info["reconstruction_error"])

    # -------------------------------------------------------------------------
    # 3. WeightMatrixDecomposer Tests
    # -------------------------------------------------------------------------
    def test_svd_decomposition(self):
        decomposer = WeightMatrixDecomposer(output_dir=self.temp_dir)
        W = np.random.randn(64, 64).astype(np.float32)

        svd_res = decomposer.decompose_svd(W, rank=16)
        self.assertEqual(svd_res["rank"], 16)
        self.assertEqual(len(svd_res["singular_values"]), 64)
        self.assertEqual(len(svd_res["singular_values_truncated"]), 16)
        self.assertGreater(svd_res["energy_retained"], 0.0)
        self.assertLess(svd_res["reconstruction_error"], 1.0)
        self.assertGreater(svd_res["param_reduction_ratio"], 0.0)

    def test_svd_energy_threshold(self):
        decomposer = WeightMatrixDecomposer(output_dir=self.temp_dir)
        # Create low rank matrix plus small noise
        u = np.random.randn(64, 8)
        v = np.random.randn(8, 64)
        W = (u @ v).astype(np.float32) + 0.001 * np.random.randn(64, 64).astype(np.float32)

        svd_res = decomposer.decompose_svd(W, energy_threshold=0.99)
        self.assertLessEqual(svd_res["rank"], 12)
        self.assertGreaterEqual(svd_res["energy_retained"], 0.99)

    def test_cur_decomposition(self):
        decomposer = WeightMatrixDecomposer(output_dir=self.temp_dir)
        W = np.random.randn(48, 48).astype(np.float32)

        cur_res = decomposer.decompose_cur(W, rank=8, num_cols=16, num_rows=16)
        self.assertEqual(cur_res["num_cols"], 16)
        self.assertEqual(cur_res["num_rows"], 16)
        self.assertEqual(cur_res["C_shape"], [48, 16])
        self.assertEqual(cur_res["R_shape"], [16, 48])
        self.assertIn("reconstruction_error", cur_res)

    def test_rpca_decomposition(self):
        decomposer = WeightMatrixDecomposer(output_dir=self.temp_dir)
        # Low rank L + Sparse S
        L = (np.random.randn(32, 4) @ np.random.randn(4, 32)).astype(np.float32)
        S = np.zeros((32, 32), dtype=np.float32)
        S[5, 5] = 10.0
        S[12, 20] = -8.0
        W = L + S

        rpca_res = decomposer.decompose_rpca(W, max_iter=40, tol=1e-4)
        self.assertIn("estimated_rank_L", rpca_res)
        self.assertGreater(rpca_res["sparsity_ratio_S"], 0.8)

    def test_weight_decomposer_plots(self):
        decomposer = WeightMatrixDecomposer(output_dir=self.temp_dir)
        W = np.random.randn(32, 32).astype(np.float32)
        svd_res = decomposer.decompose_svd(W, rank=8)

        p1 = decomposer.plot_spectrum_and_energy(svd_res["singular_values"], rank=8, save=True)
        p2 = decomposer.plot_matrix_reconstruction_diff(W, svd_res["reconstructed_W"], save=True)
        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))

    # -------------------------------------------------------------------------
    # 4. LoRAAdapterStudio Tests
    # -------------------------------------------------------------------------
    def test_lora_rank_spectrum(self):
        studio = LoRAAdapterStudio(output_dir=self.temp_dir)
        r, d_in, d_out = 8, 64, 64
        A = np.random.randn(r, d_in).astype(np.float32)
        B = np.random.randn(d_out, r).astype(np.float32)

        res = studio.analyze_rank_spectrum(A, B, alpha=16.0)
        self.assertEqual(res["rank"], 8)
        self.assertAlmostEqual(res["scaling"], 2.0)
        self.assertEqual(len(res["singular_values_adapter"]), 8)
        self.assertEqual(len(res["singular_values"]), 64)
        self.assertGreater(res["effective_rank"], 0.0)
        self.assertGreater(res["norm_delta_w"], 0.0)

    def test_lora_delta_w_magnitudes(self):
        studio = LoRAAdapterStudio(output_dir=self.temp_dir)
        r, d_in, d_out = 4, 32, 32
        W_base = np.random.randn(d_out, d_in).astype(np.float32)
        A = np.random.randn(r, d_in).astype(np.float32) * 0.1
        B = np.random.randn(d_out, r).astype(np.float32) * 0.1

        mag_res = studio.compute_delta_w_magnitudes(W_base, A, B, alpha=4.0)
        self.assertIn("relative_magnitude", mag_res)
        self.assertEqual(len(mag_res["row_intensities"]), d_out)
        self.assertEqual(len(mag_res["col_intensities"]), d_in)

    def test_lora_gradient_dynamics(self):
        studio = LoRAAdapterStudio(output_dir=self.temp_dir)
        r, d_in, d_out = 4, 16, 16
        W_base = np.random.randn(d_out, d_in).astype(np.float32)
        A = np.random.randn(r, d_in).astype(np.float32)
        B = np.random.randn(d_out, r).astype(np.float32)
        dY = np.random.randn(d_out, 1).astype(np.float32)
        X = np.random.randn(d_in, 1).astype(np.float32)

        dyn_res = studio.simulate_low_rank_gradient_dynamics(W_base, A, B, dY, X, lr=1e-3, alpha=4.0)
        self.assertIn("grad_A_norm", dyn_res)
        self.assertIn("grad_B_norm", dyn_res)
        self.assertIn("subspace_alignment", dyn_res)
        self.assertIn("updated_delta_norm", dyn_res)

    def test_lora_initialization_comparison(self):
        studio = LoRAAdapterStudio(output_dir=self.temp_dir)
        comp_res = studio.compare_initializations(d_out=32, d_in=32, rank=4)
        self.assertAlmostEqual(comp_res["standard_init"]["norm_delta_w"], 0.0)
        self.assertGreater(comp_res["svd_init"]["energy_captured"], 0.0)

    def test_lora_plot(self):
        studio = LoRAAdapterStudio(output_dir=self.temp_dir)
        s_vals = [5.0, 3.2, 1.8, 0.9]
        dw = np.random.randn(16, 16).astype(np.float32)
        plot_path = studio.plot_lora_spectrum(s_vals, delta_w=dw, save=True)
        self.assertTrue(os.path.exists(plot_path))

    # -------------------------------------------------------------------------
    # 5. Registry Verification
    # -------------------------------------------------------------------------
    def test_registry_registration(self):
        self.assertIn("ffn_neuron_explorer", VISUALIZER_REGISTRY)
        self.assertIn("kv_cache_monitor", VISUALIZER_REGISTRY)
        self.assertIn("weight_matrix_decomposer", VISUALIZER_REGISTRY)
        self.assertIn("lora_adapter_studio", VISUALIZER_REGISTRY)


if __name__ == "__main__":
    unittest.main()
