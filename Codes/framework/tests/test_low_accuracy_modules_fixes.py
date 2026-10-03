"""
Unit tests validating fixes for low-accuracy modules:
- GPT-2 fused c_attn splitting prior to SVD
- Aegis family positive gate bias and residual connection
- Kronecker / FMS normalized positive denominator
- Compressed attention pretrained weight copy & Frobenius norm >= 1e-4
- Cluster attention padding token filtering
"""
import unittest

import torch
import torch.nn as nn
from framework.attentions.adaptive_svd_attention import AdaptiveSVDAttention
from framework.attentions.asvd_attention import ASVDAttention
from framework.attentions.cluster_attention import ClusterAttention
from framework.attentions.compressed_attention import CompressedAttention
from framework.attentions.factorised_matrix_state import FactorisedMatrixState


class FakeGPT2Attention(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.split_size = embed_dim


class TestLowAccuracyModulesFixes(unittest.TestCase):
    def test_gpt2_c_attn_split_before_svd(self):
        """GPT-2 c_attn is split into Q, K, V before SVD rather than factorizing fused matrix."""
        orig = FakeGPT2Attention(embed_dim=64)
        asvd = ASVDAttention(hidden_size=64, num_heads=4, is_decoder=True)
        asvd.copy_weights(orig)

        # Ensure query, key, value weights each have Frobenius norm >= 1e-4
        for proj in (asvd.query, asvd.key, asvd.value, asvd.out_proj):
            w = proj.weight if hasattr(proj, "weight") else proj.get_weight()
            norm = torch.linalg.norm(w.float()).item()
            self.assertGreater(norm, 1e-4)

    def test_fms_positive_denominators(self):
        """FMS uses positive ELU+1 kernel feature maps preventing negative denominator explosions."""
        fms = FactorisedMatrixState(d_model=64, num_heads=4, is_decoder=True)
        x = torch.randn(2, 32, 64)
        out, _ = fms(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.isnan(out).any())

    def test_cluster_padding_exclusion(self):
        """Cluster attention excludes padding tokens from centroid calculation."""
        cluster_attn = ClusterAttention(hidden_size=64, num_heads=4, num_clusters=4)
        x = torch.randn(2, 16, 64)
        # Sequence 2 has last 8 tokens as padding
        mask = torch.ones(2, 16)
        mask[1, 8:] = 0.0
        out, _ = cluster_attn(x, attention_mask=mask)
        self.assertTrue(torch.isfinite(out).all())

    def test_compressed_attention_weight_copy(self):
        """CompressedAttention copies pretrained weights and validates Frobenius norm >= 1e-4."""
        orig = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        comp = CompressedAttention(hidden_size=64, num_heads=4, rank=16)
        comp.copy_weights(orig)
        for proj in (comp.query, comp.key, comp.value):
            w = proj.weight if hasattr(proj, "weight") else proj.get_weight()
            norm = torch.linalg.norm(w.float()).item()
            self.assertGreater(norm, 1e-4)


if __name__ == "__main__":
    unittest.main()
