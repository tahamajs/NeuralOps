"""
Comprehensive test suite for Attention Dashboard and Transformer Studios REST APIs.
Covers:
- Attention Canvas (Zones 1-4)
- Interpretability & Analytics (Zones 5-9)
- Representation Studios (3D Embedding, RoPE, Residual, LayerNorm)
- Computation Studios (FFN Universe, KV Cache, Weight Decomposer, LoRA)
- Optimization Studios (Circuit Attribution, Logit Lens, Loss Landscape, Compression)
- Multi-Model Architecture Comparison API
"""

import json
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient


class AttentionDashboardAPITests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="researcher", password="password123")
        self.token = Token.objects.create(user=self.user)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token.key}")

    def test_attention_canvas_api(self):
        """Test GET /api/attention-visualization/canvas/"""
        response = self.client.get(
            "/api/attention-visualization/canvas/",
            {"layer": 0, "head": 0, "model": "svd_linear"}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("matrix", data)
        self.assertIn("tokens", data)
        self.assertIn("model_view", data)
        self.assertIn("sankey_flow", data)
        self.assertIn("qk_projections", data)
        self.assertIn("math_breakdown", data)
        self.assertEqual(len(data["matrix"]), len(data["tokens"]))

    def test_interpretability_suite_api(self):
        """Test GET /api/attention-visualization/interpretability/"""
        response = self.client.get(
            "/api/attention-visualization/interpretability/",
            {"layer": 2, "head": 1, "model": "mamba_svd_flash"}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("head_3d_metrics", data)
        self.assertIn("ablation_sensitivity", data)
        self.assertIn("circuit_svd", data)
        self.assertIn("logit_lens", data)
        self.assertIn("semantic_pathways", data)

    def test_representation_studios_api(self):
        """Test GET /api/attention-visualization/studios/representation/"""
        response = self.client.get(
            "/api/attention-visualization/studios/representation/",
            {"layer": 1, "reduction": "pca"}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("embedding_cloud", data)
        self.assertIn("rope_frequencies", data)
        self.assertIn("residual_stream", data)
        self.assertIn("layernorm_hypersphere", data)

    def test_computation_studios_api(self):
        """Test GET /api/attention-visualization/studios/computation/"""
        response = self.client.get(
            "/api/attention-visualization/studios/computation/",
            {"layer": 3, "method": "svd"}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("ffn_activations", data)
        self.assertIn("kv_cache_monitor", data)
        self.assertIn("weight_decomposition", data)
        self.assertIn("lora_adapter", data)

    def test_optimization_studios_api(self):
        """Test GET /api/attention-visualization/studios/optimization/"""
        response = self.client.get(
            "/api/attention-visualization/studios/optimization/",
            {"target_token_idx": 4}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("circuit_attribution", data)
        self.assertIn("logit_lens_timeline", data)
        self.assertIn("loss_landscape_3d", data)
        self.assertIn("compression_pareto", data)

    def test_architecture_comparison_api(self):
        """Test POST /api/attention-visualization/compare/"""
        payload = {
            "models": ["svd_linear", "flash3", "gqa", "full"],
            "dataset": "wikitext2",
            "seq_lens": [512, 1024, 2048, 8192]
        }
        response = self.client.post(
            "/api/attention-visualization/compare/",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("comparison_matrix", data)
        self.assertIn("recommendations", data)
        self.assertEqual(len(data["comparison_matrix"]), 4)
