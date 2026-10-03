"""
Unit tests for Attention Canvas Suite (Zones 1-4 Engine)
=========================================================
Verifies:
- Head View extraction, entropy, concentration, top pairs, and SVG generation
- Model View all-layers x all-heads metrics, diversity, importance, and classification
- Multi-layer Sankey attention flow graph generation
- Q/K joint 2D embedding (PCA, t-SNE, SVD)
- Live Math KaTeX breakdowns for Softmax MHA, SVD Attention, Linear Attention, and GQA
- End-to-end full dashboard payload serialization
"""

import json
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from framework.visualization.attention_canvas_suite import (
    AttentionCanvasSuite, HeadViewData, LiveMathBreakdown, ModelViewData,
    QKEmbeddingData, SankeyData, classify_head_specialization,
    compute_model_view_metrics, compute_qk_joint_embedding, extract_head_view,
    generate_head_heatmap_svg, generate_live_math_breakdown,
    generate_paper_latex_section, generate_sankey_flow_data)


class TestAttentionCanvasSuite(unittest.TestCase):
    """Comprehensive test suite for Attention Canvas Suite visualization engine."""

    def setUp(self):
        self.num_layers = 4
        self.num_heads = 4
        self.seq_len = 8
        self.d_k = 32
        self.tokens = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]

        # Deterministic synthetic attention tensor [L, H, N, N]
        np.random.seed(42)
        raw = np.random.exponential(scale=1.0, size=(self.num_layers, self.num_heads, self.seq_len, self.seq_len)).astype(np.float32)
        # Add diagonal bias
        for l in range(self.num_layers):
            for h in range(self.num_heads):
                for i in range(self.seq_len):
                    raw[l, h, i, i] += 3.0
                    if i > 0:
                        raw[l, h, i, i - 1] += 1.5
        exp_raw = np.exp(raw - np.max(raw, axis=-1, keepdims=True))
        self.attn_tensor = exp_raw / np.sum(exp_raw, axis=-1, keepdims=True)

        self.query_matrix = np.random.randn(self.seq_len, self.d_k).astype(np.float32)
        self.key_matrix = np.random.randn(self.seq_len, self.d_k).astype(np.float32)

    def test_extract_head_view(self):
        """Test extraction of single head attention matrix and statistics."""
        head_data = extract_head_view(
            attention_matrix=self.attn_tensor,
            tokens=self.tokens,
            layer_idx=1,
            head_idx=2,
            model_name="BERT-base",
            attention_type="SVD Attention",
            generate_svg=True,
        )
        self.assertIsInstance(head_data, HeadViewData)
        self.assertEqual(head_data.layer_idx, 1)
        self.assertEqual(head_data.head_idx, 2)
        self.assertEqual(len(head_data.matrix), self.seq_len)
        self.assertEqual(len(head_data.matrix[0]), self.seq_len)
        self.assertGreater(head_data.mean_entropy, 0.0)
        self.assertGreater(head_data.concentration_top3, 0.0)
        self.assertLessEqual(head_data.concentration_top3, 1.0)
        self.assertGreater(head_data.diagonal_ratio, 0.0)
        self.assertTrue(len(head_data.top_attended_pairs) > 0)
        self.assertIsNotNone(head_data.svg_heatmap)
        self.assertIn("<svg", head_data.svg_heatmap)

    def test_compute_model_view_metrics(self):
        """Test model-wide metrics across all layers and heads."""
        model_data = compute_model_view_metrics(
            attention_tensor=self.attn_tensor,
            tokens=self.tokens,
            include_thumbnails=True,
            thumbnail_size=4,
        )
        self.assertIsInstance(model_data, ModelViewData)
        self.assertEqual(model_data.num_layers, self.num_layers)
        self.assertEqual(model_data.num_heads, self.num_heads)
        self.assertEqual(len(model_data.heads), self.num_layers * self.num_heads)
        self.assertEqual(len(model_data.layer_entropies), self.num_layers)
        self.assertEqual(len(model_data.layer_importances), self.num_layers)

        for hm in model_data.heads:
            self.assertGreaterEqual(hm.importance, 0.0)
            self.assertGreaterEqual(hm.diversity, 0.0)
            self.assertGreaterEqual(hm.entropy, 0.0)
            self.assertIn(hm.specialization, [
                "Positional Head", "Induction Head", "Semantic Head", "Delimiter Head", "Broad Context Head"
            ])
            self.assertIsNotNone(hm.thumbnail)
            self.assertEqual(len(hm.thumbnail), 4)

    def test_generate_sankey_flow_data(self):
        """Test multi-layer Sankey attention flow graph generation."""
        sankey = generate_sankey_flow_data(
            attention_tensor=self.attn_tensor,
            tokens=self.tokens,
            top_k_flows=25,
            threshold=0.01,
        )
        self.assertIsInstance(sankey, SankeyData)
        self.assertGreater(len(sankey.nodes), 0)
        self.assertGreater(len(sankey.links), 0)
        self.assertLessEqual(len(sankey.links), 25)

        # Check input and output node types exist
        node_types = {n.node_type for n in sankey.nodes}
        self.assertIn("input", node_types)
        self.assertIn("head", node_types)
        self.assertIn("output", node_types)

    def test_compute_qk_joint_embedding(self):
        """Test Query/Key 2D joint embedding generation (PCA & t-SNE)."""
        # PCA
        pca_emb = compute_qk_joint_embedding(
            query_matrix=self.query_matrix,
            key_matrix=self.key_matrix,
            tokens=self.tokens,
            method="pca",
        )
        self.assertIsInstance(pca_emb, QKEmbeddingData)
        self.assertEqual(pca_emb.method, "PCA")
        self.assertEqual(len(pca_emb.points), 2 * self.seq_len)
        self.assertEqual(len(pca_emb.tokens), self.seq_len)
        self.assertIsNotNone(pca_emb.explained_variance_ratio)

        # t-SNE
        tsne_emb = compute_qk_joint_embedding(
            query_matrix=self.query_matrix,
            key_matrix=self.key_matrix,
            tokens=self.tokens,
            method="tsne",
        )
        self.assertEqual(tsne_emb.method, "TSNE")
        self.assertEqual(len(tsne_emb.points), 2 * self.seq_len)

    def test_generate_live_math_breakdown(self):
        """Test KaTeX mathematical breakdowns across different attention architectures."""
        types_to_test = ["svd", "mha", "linear", "gqa"]
        for attn_type in types_to_test:
            breakdown = generate_live_math_breakdown(
                attention_type=attn_type,
                q_vec=self.query_matrix[0],
                k_vec=self.key_matrix[1],
                d_k=self.d_k,
                tokens=self.tokens,
            )
            self.assertIsInstance(breakdown, LiveMathBreakdown)
            self.assertGreater(len(breakdown.steps), 0)
            self.assertIn("Q", breakdown.variable_color_map)
            self.assertIn("time_complexity", breakdown.complexity_table)
            self.assertIn("raw_dot_product", breakdown.numerical_inspection)
            self.assertEqual(breakdown.numerical_inspection["query_index"], 0)
            self.assertEqual(breakdown.numerical_inspection["key_index"], 1)

    def test_attention_canvas_suite_full_export(self):
        """Test end-to-end payload export and JSON serializability."""
        suite = AttentionCanvasSuite(
            model_name="BERT-base-uncased",
            attention_type="SVD Linear Attention (rank=64)",
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        payload = suite.export_full_dashboard_payload(
            tokens=self.tokens,
            selected_layer=2,
            selected_head=1,
            attention_tensor=self.attn_tensor,
            query_matrix=self.query_matrix,
            key_matrix=self.key_matrix,
        )
        self.assertIn("context", payload)
        self.assertIn("head_view", payload)
        self.assertIn("model_view", payload)
        self.assertIn("sankey_flow", payload)
        self.assertIn("qk_embeddings", payload)
        self.assertIn("live_math", payload)
        self.assertIn("paper_export_latex", payload)

        # Ensure entire payload is JSON serializable
        json_str = json.dumps(payload)
        self.assertGreater(len(json_str), 100)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["context"]["selected_layer"], 2)
        self.assertEqual(parsed["context"]["selected_head"], 1)

    def test_torch_tensor_support(self):
        """Test seamless compatibility with PyTorch tensors."""
        torch_tensor = torch.from_numpy(self.attn_tensor)
        head_data = extract_head_view(torch_tensor, self.tokens, layer_idx=0, head_idx=0)
        self.assertEqual(len(head_data.matrix), self.seq_len)

        model_data = compute_model_view_metrics(torch_tensor, self.tokens)
        self.assertEqual(model_data.num_layers, self.num_layers)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAttentionCanvasSuite)
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

