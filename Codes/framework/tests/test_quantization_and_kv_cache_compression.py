"""
Comprehensive Unit Tests for Quantization & KV-Cache Compression Engine modules.
Tests:
1. astro_quantizer (AstroQuantizer, AstroLinear, STEFakeQuantize)
2. faq_quantizer (FAQQuantizer, FAQLinear)
3. coopq_quantizer (CoopQQuantizer)
4. kmeans_quantize (KMeansCompressor, KMeansLinear, fast_kmeans_1d)
5. delta_kv (DeltaKVCompressor, DeltaKVStore)
6. hybridkv_compressor (HybridKVCompressor, HybridKVCache)
7. kvtc_compressor (KVTCCompressor, KVTCStore, SimpleEntropyCoder)
8. cachenotes_compressor (CacheNotesCompressor, CacheNotesKVCache, CompressionPlanningTokens)
9. zerokv_neo (ZeroKVNeoCompressor, ZeroKVNeoCache)
10. modegpt (MoDeGPTCompressor)
11. huffman_coder (HuffmanCoder, HuffmanCompressor)
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Dict, Any

import torch
import torch.nn as nn

# Ensure framework is in path
_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

from framework.compression_engine.astro_quantizer import (
    AstroQuantizer,
    AstroLinear,
    STEFakeQuantize,
    quantize_dequantize_tensor,
)
from framework.compression_engine.faq_quantizer import FAQQuantizer, FAQLinear
from framework.compression_engine.coopq_quantizer import CoopQQuantizer
from framework.compression_engine.kmeans_quantize import (
    KMeansCompressor,
    KMeansLinear,
    fast_kmeans_1d,
)
from framework.compression_engine.delta_kv import DeltaKVCompressor, DeltaKVStore
from framework.compression_engine.hybridkv_compressor import (
    HybridKVCompressor,
    HybridKVCache,
)
from framework.compression_engine.kvtc_compressor import (
    KVTCCompressor,
    KVTCStore,
    SimpleEntropyCoder,
)
from framework.compression_engine.cachenotes_compressor import (
    CacheNotesCompressor,
    CacheNotesKVCache,
    CompressionPlanningTokens,
)
from framework.compression_engine.zerokv_neo_compressor import (
    ZeroKVNeoCompressor,
    ZeroKVNeoCache,
)
from framework.compression_engine.modegpt_compressor import MoDeGPTCompressor
from framework.compression_engine.huffman_coder import HuffmanCoder, HuffmanCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class ToyTransformerBlock(nn.Module):
    """Toy multi-layer transformer block for testing compression engines."""

    def __init__(self, d_model: int = 64, d_ff: int = 128):
        super().__init__()
        self.d_model = d_model
        # Attention projection pairs
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        # MLP projections
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_lin(x)
        k = self.k_lin(x)
        v = self.v_lin(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model**0.5)
        attn_weights = torch.softmax(attn, dim=-1)
        attn_out = self.out_lin(torch.matmul(attn_weights, v))
        x = x + attn_out
        mlp_out = self.fc2(self.act(self.fc1(x)))
        return x + mlp_out


class TestQuantizationAndKVCacheCompression(unittest.TestCase):
    """Comprehensive test suite for all 11 compression and quantization engines."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.batch_size = 2
        self.seq_len = 16
        self.dummy_x = torch.randn(self.batch_size, self.seq_len, self.d_model)

    # ----------------------------------------------------------------------
    # 1. AstroQuantizer Tests
    # ----------------------------------------------------------------------
    def test_astro_quantizer_int8_and_int4(self):
        """Test AstroQuantizer INT8 and INT4 quantization and output tensor validity."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = AstroQuantizer(model, bits=8, symmetric=True)
        comp_model, stats = compressor.compress()

        self.assertIn("quantization_bits", stats)
        self.assertEqual(stats["quantization_bits"], 8)
        self.assertIn("compressed_parameters", stats)

        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(out).any())

        # Test INT4 with ASTRO factory
        model4 = ToyTransformerBlock(d_model=self.d_model)
        factory_fn = COMPRESSOR_REGISTRY.get("astro")
        self.assertIsNotNone(factory_fn)

        class Config:
            astro_bits = 4
            astro_symmetric = False

        comp_model4 = factory_fn(model4, Config())
        out4 = comp_model4(self.dummy_x)
        self.assertEqual(out4.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_astro_linear_ste_backward(self):
        """Test AstroLinear Straight-Through Estimator (STE) gradient backpropagation."""
        linear = nn.Linear(32, 32)
        astro_layer = AstroLinear.from_linear(linear, bits=4, symmetric=True)
        astro_layer.train()

        x = torch.randn(2, 8, 32, requires_grad=True)
        out = astro_layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(astro_layer.weight.grad)
        self.assertGreater(astro_layer.weight.grad.abs().sum().item(), 0.0)

    # ----------------------------------------------------------------------
    # 2. FAQQuantizer Tests
    # ----------------------------------------------------------------------
    def test_faq_quantizer_calibration_and_forward(self):
        """Test FAQQuantizer activation calibration, INT8/INT4 quantization, and forward execution."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = FAQQuantizer(model, bits=8, alpha=0.5)
        comp_model, stats = compressor.compress()

        self.assertEqual(stats["compressor_name"], "faq")
        self.assertEqual(stats["quantization_bits"], 8)
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(out).any())

    def test_faq_linear_ste(self):
        """Test FAQLinear layer forward pass and STE gradient propagation."""
        base_lin = nn.Linear(32, 32)
        act_scale = torch.rand(32) + 0.1
        faq_layer = FAQLinear.from_linear(base_lin, act_scale=act_scale, bits=4)
        faq_layer.train()

        x = torch.randn(2, 4, 32, requires_grad=True)
        out = faq_layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    # ----------------------------------------------------------------------
    # 3. CoopQQuantizer Tests
    # ----------------------------------------------------------------------
    def test_coopq_quantizer(self):
        """Test CoopQ game-theoretic bit allocation across layers."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = CoopQQuantizer(model, bits=4, min_bits=2, max_bits=8)
        comp_model, stats = compressor.compress()

        self.assertEqual(stats["compressor_name"], "coopq")
        self.assertIn("layer_bit_allocations", stats)
        self.assertIn("average_allocated_bits", stats)
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 4. KMeansQuantizer Tests
    # ----------------------------------------------------------------------
    def test_kmeans_compressor_and_ste(self):
        """Test KMeansCompressor, fast 1D k-means clustering, and KMeansLinear STE backward pass."""
        # 1D k-means algorithm verification
        t = torch.randn(100, 50)
        centroids, indices = fast_kmeans_1d(t, n_clusters=16)
        self.assertEqual(centroids.shape, (16,))
        self.assertEqual(indices.shape, (100, 50))

        # Model compression test
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = KMeansCompressor(model, n_clusters=16)
        comp_model, stats = compressor.compress()

        self.assertEqual(stats["compressor_name"], "kmeans")
        self.assertEqual(stats["n_clusters"], 16)
        self.assertEqual(stats["effective_bits_per_weight"], 4)

        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

        # Test STE backprop through KMeansLinear
        lin = nn.Linear(32, 32)
        km_lin = KMeansLinear(lin, n_clusters=16)
        km_lin.train()
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = km_lin(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(km_lin.centroids.grad)
        self.assertGreater(km_lin.centroids.grad.abs().sum().item(), 0.0)

    # ----------------------------------------------------------------------
    # 5. DeltaKV Tests
    # ----------------------------------------------------------------------
    def test_delta_kv_dynamic_sequence_lengths(self):
        """Test DeltaKVStore handling dynamic sequence lengths (prefill + autoregressive steps)."""
        store = DeltaKVStore(max_refs=64, topk=4, compression_ratio=0.29)
        B, H, D = 2, 4, 32

        # Step 1: Long prompt prefill (N=64)
        k_prefill = torch.randn(B, H, 64, D)
        v_prefill = torch.randn(B, H, 64, D)
        k_c, v_c = store.compress(k_prefill, v_prefill)
        self.assertEqual(k_c.shape, (B, H, 64, D))
        self.assertEqual(v_c.shape, (B, H, 64, D))

        # Step 2: Autoregressive single token generation (N=1) for 10 steps
        for step in range(10):
            k_step = torch.randn(B, H, 1, D)
            v_step = torch.randn(B, H, 1, D)
            k_sc, v_sc = store.compress(k_step, v_step)
            self.assertEqual(k_sc.shape, (B, H, 1, D))
            self.assertEqual(v_sc.shape, (B, H, 1, D))

        # Step 3: Arbitrary chunk update (N=17)
        k_chunk = torch.randn(B, H, 17, D)
        v_chunk = torch.randn(B, H, 17, D)
        k_cc, v_cc = store.compress(k_chunk, v_chunk)
        self.assertEqual(k_cc.shape, (B, H, 17, D))

    def test_delta_kv_compressor(self):
        """Test DeltaKVCompressor model weight compression."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = DeltaKVCompressor(model, rank=16, sparsity=0.29)
        comp_model, stats = compressor.compress()
        self.assertEqual(stats["compressor_name"], "delta_kv")
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 6. HybridKV Tests
    # ----------------------------------------------------------------------
    def test_hybridkv_dynamic_cache(self):
        """Test HybridKVCache dynamic eviction with attention sinks and sliding window."""
        cache = HybridKVCache(max_cache_len=32, num_sinks=4, recent_window=8)
        B, H, D = 2, 4, 32

        # Add initial 20 tokens (within capacity)
        k1 = torch.randn(B, H, 20, D)
        v1 = torch.randn(B, H, 20, D)
        k_out1, v_out1 = cache.update_and_compress(k1, v1)
        self.assertEqual(k_out1.shape, (B, H, 20, D))

        # Add 30 more tokens (total 50 > max 32)
        k2 = torch.randn(B, H, 30, D)
        v2 = torch.randn(B, H, 30, D)
        k_out2, v_out2 = cache.update_and_compress(k2, v2)
        self.assertEqual(k_out2.shape, (B, H, 32, D))
        self.assertEqual(v_out2.shape, (B, H, 32, D))

        # Decoding step (N=1)
        k3 = torch.randn(B, H, 1, D)
        v3 = torch.randn(B, H, 1, D)
        k_out3, v_out3 = cache.update_and_compress(k3, v3)
        self.assertEqual(k_out3.shape, (B, H, 32, D))

    def test_hybridkv_compressor(self):
        """Test HybridKVCompressor weight pruning."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = HybridKVCompressor(model, budget=0.4, static_ratio=0.7)
        comp_model, stats = compressor.compress()
        self.assertEqual(stats["compressor_name"], "hybridkv")
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 7. KVTC Tests
    # ----------------------------------------------------------------------
    def test_kvtc_compressor_and_store(self):
        """Test KVTCCompressor and KVTCStore runtime transform coding."""
        # Test KVTC runtime cache
        store = KVTCStore(bits=4, pca_energy=0.95)
        B, H, N, D = 2, 4, 32, 16
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        k_hat, v_hat, stats = store.compress_kv_cache(k, v)
        self.assertEqual(k_hat.shape, (B, H, N, D))
        self.assertIn("mean_entropy_bits", stats)
        self.assertLessEqual(stats["mean_entropy_bits"], 5.0)

        # Test model compressor
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = KVTCCompressor(model, bits=4, pca_energy=0.95)
        comp_model, comp_stats = compressor.compress()
        self.assertEqual(comp_stats["compressor_name"], "kvtc")
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 8. CacheNotes Tests
    # ----------------------------------------------------------------------
    def test_cachenotes_cache_and_compressor(self):
        """Test CacheNotes dynamic CPT cache and model compressor."""
        # Test CPT dynamic cache
        cache = CacheNotesKVCache(max_cache_len=24, task_type="code", keep_ratio=0.3, num_sinks=4)
        B, H, D = 2, 4, 32
        k = torch.randn(B, H, 40, D)
        v = torch.randn(B, H, 40, D)
        k_out, v_out = cache.update_and_compress(k, v)
        self.assertEqual(k_out.shape, (B, H, 24, D))

        # Single step decode
        k_step = torch.randn(B, H, 1, D)
        v_step = torch.randn(B, H, 1, D)
        k_out2, _ = cache.update_and_compress(k_step, v_step)
        self.assertEqual(k_out2.shape, (B, H, 24, D))

        # Model compressor
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = CacheNotesCompressor(model, task_type="qa", keep_ratio=0.4)
        comp_model, stats = compressor.compress()
        self.assertEqual(stats["compressor_name"], "cachenotes")
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 9. ZeroKV-Neo Tests
    # ----------------------------------------------------------------------
    def test_zerokv_neo_cache_and_compressor(self):
        """Test ZeroKVNeoCache bounded memory eviction and model compressor."""
        # Dynamic cache
        cache = ZeroKVNeoCache(max_cache_len=20, num_attention_sinks=4, recent_window=6)
        B, H, D = 2, 4, 32

        # 50 tokens -> evicted to 20
        k = torch.randn(B, H, 50, D)
        v = torch.randn(B, H, 50, D)
        k_c, v_c = cache.update_and_compress(k, v)
        self.assertEqual(k_c.shape, (B, H, 20, D))
        self.assertEqual(v_c.shape, (B, H, 20, D))

        # Model compressor
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = ZeroKVNeoCompressor(model, keep_ratio=0.3, num_attention_sinks=4)
        comp_model, stats = compressor.compress()
        self.assertEqual(stats["compressor_name"], "zerokv_neo")
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 10. MoDeGPT Tests
    # ----------------------------------------------------------------------
    def test_modegpt_compressor(self):
        """Test MoDeGPT joint SVD modular decomposition on projection pairs."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = MoDeGPTCompressor(model, rank=16, method="svd")
        comp_model, stats = compressor.compress()
        self.assertEqual(stats["compressor_name"], "modegpt")
        self.assertEqual(stats["rank"], 16)
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 11. HuffmanCoder Tests
    # ----------------------------------------------------------------------
    def test_huffman_coder_and_compressor(self):
        """Test HuffmanCoder bitstream packing, lossless decoding, and HuffmanCompressor."""
        # Tensor encode and decode
        t = torch.randn(32, 32)
        packed_bytes, code_map, meta = HuffmanCoder.encode(t, quantize_bits=8)
        self.assertGreater(len(packed_bytes), 0)
        self.assertIn("entropy", meta)
        self.assertIn("coding_efficiency_pct", meta)

        decoded = HuffmanCoder.decode(packed_bytes, code_map, meta)
        self.assertEqual(decoded.shape, (32, 32))
        # Quantization reconstruction error check
        mse = torch.mean((t - decoded) ** 2).item()
        self.assertLess(mse, 0.05)

        # Huffman model compressor
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = HuffmanCompressor(model, quantize_bits=8)
        comp_model, stats = compressor.compress()
        self.assertEqual(stats["compressor_name"], "huffman")
        self.assertIn("huffman_compression_ratio", stats)
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))


if __name__ == "__main__":
    unittest.main()
