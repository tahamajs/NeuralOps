"""
Comprehensive Unit Tests for Quantization & KV-Cache Compression Engine modules.
Tests all 17 modules:
1. astro_quantizer (AstroQuantizer, AstroLinear, STEFakeQuantize, astro_factory)
2. faq_quantizer (FAQQuantizer, FAQLinear, faq_factory)
3. coopq_quantizer (CoopQQuantizer, coopq_factory)
4. kmeans_quantize (KMeansCompressor, KMeansLinear, fast_kmeans_1d, kmeans_factory)
5. delta_kv (DeltaKVCompressor, DeltaKVStore, delta_kv_factory)
6. hybridkv_compressor (HybridKVCompressor, HybridKVCache, hybridkv_factory)
7. kvtc_compressor (KVTCCompressor, KVTCStore, SimpleEntropyCoder, kvtc_factory)
8. cachenotes_compressor (CacheNotesCompressor, CacheNotesKVCache, CompressionPlanningTokens, cachenotes_factory)
9. zerokv_neo & zerokv_neo_compressor (ZeroKVNeoCompressor, ZeroKVNeoCache, zerokv_neo_factory)
10. modegpt & modegpt_compressor (MoDeGPTCompressor, modegpt_factory)
11. huffman_coder (HuffmanCoder, HuffmanCompressor, huffman_factory)
12. uniql_compressor (UniQLCompressor, uniql_factory)
13. sdrm_compressor (SDRMCompressor, sdrm_compressor_factory)
14. enlighten_compressor (ENLightenCompressor, enlighten_factory)
15. nli_softmax (NLISoftmaxFunction, nli_softmax, NLISoftmax, NLISoftmaxCompressor, nli_softmax_factory)
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict

import torch
import torch.nn as nn

# Ensure framework is in path
_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

from framework.compression_engine.astro_quantizer import (
    AstroLinear, AstroQuantizer, STEFakeQuantize, astro_factory,
    quantize_dequantize_tensor)
from framework.compression_engine.cachenotes_compressor import (
    CacheNotesCompressor, CacheNotesKVCache, CompressionPlanningTokens,
    cachenotes_factory)
from framework.compression_engine.coopq_quantizer import (CoopQQuantizer,
                                                          coopq_factory)
from framework.compression_engine.delta_kv import (DeltaKVCompressor,
                                                   DeltaKVStore,
                                                   delta_kv_factory)
from framework.compression_engine.enlighten_compressor import (
    ENLightenCompressor, enlighten_factory)
from framework.compression_engine.faq_quantizer import (FAQLinear,
                                                        FAQQuantizer,
                                                        faq_factory)
from framework.compression_engine.huffman_coder import (HuffmanCoder,
                                                        HuffmanCompressor,
                                                        huffman_factory)
from framework.compression_engine.hybridkv_compressor import (
    HybridKVCache, HybridKVCompressor, hybridkv_factory)
from framework.compression_engine.kmeans_quantize import (KMeansCompressor,
                                                          KMeansLinear,
                                                          fast_kmeans_1d,
                                                          kmeans_factory)
from framework.compression_engine.kvtc_compressor import (KVTCCompressor,
                                                          KVTCStore,
                                                          SimpleEntropyCoder,
                                                          kvtc_factory)
from framework.compression_engine.modegpt import (MoDeGPTCompressor,
                                                  modegpt_factory)
from framework.compression_engine.nli_softmax import (NLISoftmax,
                                                      NLISoftmaxCompressor,
                                                      NLISoftmaxFunction,
                                                      nli_softmax,
                                                      nli_softmax_factory)
from framework.compression_engine.sdrm_compressor import (
    SDRMCompressor, sdrm_compressor_factory)
from framework.compression_engine.uniql_compressor import (UniQLCompressor,
                                                           uniql_factory)
from framework.compression_engine.zerokv_neo import (ZeroKVNeoCache,
                                                     ZeroKVNeoCompressor,
                                                     zerokv_neo_factory)
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_lin(x)
        k = self.k_lin(x)
        v = self.v_lin(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model**0.5)
        attn_weights = self.softmax(attn)
        attn_out = self.out_lin(torch.matmul(attn_weights, v))
        x = x + attn_out
        mlp_out = self.fc2(self.act(self.fc1(x)))
        return x + mlp_out


class TestQuantizationAndKVCacheCompression(unittest.TestCase):
    """Comprehensive test suite for all 17 compression and quantization engines."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.batch_size = 2
        self.seq_len = 16
        self.dummy_x = torch.randn(self.batch_size, self.seq_len, self.d_model)

    # ----------------------------------------------------------------------
    # 1. AstroQuantizer Tests (FP32 & FP16 Stability)
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

        # Test INT4 with ASTRO factory and kwargs
        model4 = ToyTransformerBlock(d_model=self.d_model)
        comp_model4 = astro_factory(model4, bits=4, symmetric=False)
        out4 = comp_model4(self.dummy_x)
        self.assertEqual(out4.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_astro_fp16_numerical_stability(self):
        """Test Astro quantization numerical stability in FP16."""
        t_fp16 = torch.randn(16, 32, dtype=torch.float16)
        q_fp16 = quantize_dequantize_tensor(t_fp16, bits=4, symmetric=False)
        self.assertEqual(q_fp16.dtype, torch.float16)
        self.assertFalse(torch.isnan(q_fp16).any())
        self.assertFalse(torch.isinf(q_fp16).any())

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

        # Test factory with kwargs
        model_faq = ToyTransformerBlock(d_model=self.d_model)
        comp_model_faq = faq_factory(model_faq, bits=4, alpha=0.7)
        out_faq = comp_model_faq(self.dummy_x)
        self.assertEqual(out_faq.shape, (self.batch_size, self.seq_len, self.d_model))

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

        # Test factory with kwargs
        model_cq = ToyTransformerBlock(d_model=self.d_model)
        comp_cq = coopq_factory(model_cq, bits=6, min_bits=4, max_bits=8)
        out_cq = comp_cq(self.dummy_x)
        self.assertEqual(out_cq.shape, (self.batch_size, self.seq_len, self.d_model))

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

        # Test factory with kwargs
        model_km = ToyTransformerBlock(d_model=self.d_model)
        comp_km = kmeans_factory(model_km, n_clusters=32)
        out_km = comp_km(self.dummy_x)
        self.assertEqual(out_km.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 5. DeltaKV Tests (Dynamic Sequences & FP16/FP32)
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

        # Factory with kwargs
        model_dkv = ToyTransformerBlock(d_model=self.d_model)
        comp_dkv = delta_kv_factory(model_dkv, rank=8, sparsity=0.5)
        out_dkv = comp_dkv(self.dummy_x)
        self.assertEqual(out_dkv.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 6. HybridKV Tests (Dynamic Sequences & Multi-head Eviction)
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

        # Factory with kwargs
        model_h = ToyTransformerBlock(d_model=self.d_model)
        comp_h = hybridkv_factory(model_h, budget=0.5, static_ratio=0.6)
        out_h = comp_h(self.dummy_x)
        self.assertEqual(out_h.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 7. KVTC Tests (Transform Coding & PCA Calibration)
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

        # Dynamic decoding steps (N=1)
        k_dec = torch.randn(B, H, 1, D)
        v_dec = torch.randn(B, H, 1, D)
        k_hat_dec, v_hat_dec, _ = store.compress_kv_cache(k_dec, v_dec)
        self.assertEqual(k_hat_dec.shape, (B, H, 1, D))

        # Test model compressor
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = KVTCCompressor(model, bits=4, pca_energy=0.95)
        comp_model, comp_stats = compressor.compress()
        self.assertEqual(comp_stats["compressor_name"], "kvtc")
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

        # Factory with kwargs
        model_kvtc = ToyTransformerBlock(d_model=self.d_model)
        comp_kvtc = kvtc_factory(model_kvtc, bits=8, pca_energy=0.90)
        out_kvtc = comp_kvtc(self.dummy_x)
        self.assertEqual(out_kvtc.shape, (self.batch_size, self.seq_len, self.d_model))

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

        # Factory with kwargs
        model_cn = ToyTransformerBlock(d_model=self.d_model)
        comp_cn = cachenotes_factory(model_cn, task_type="math", keep_ratio=0.5)
        out_cn = comp_cn(self.dummy_x)
        self.assertEqual(out_cn.shape, (self.batch_size, self.seq_len, self.d_model))

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

        # Autoregressive decoding token update (N=1)
        k_step = torch.randn(B, H, 1, D)
        v_step = torch.randn(B, H, 1, D)
        k_c2, v_c2 = cache.update_and_compress(k_step, v_step)
        self.assertEqual(k_c2.shape, (B, H, 20, D))

        # Model compressor
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = ZeroKVNeoCompressor(model, keep_ratio=0.3, num_attention_sinks=4)
        comp_model, stats = compressor.compress()
        self.assertEqual(stats["compressor_name"], "zerokv_neo")
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

        # Factory with kwargs
        model_zk = ToyTransformerBlock(d_model=self.d_model)
        comp_zk = zerokv_neo_factory(model_zk, keep_ratio=0.4, num_attention_sinks=2)
        out_zk = comp_zk(self.dummy_x)
        self.assertEqual(out_zk.shape, (self.batch_size, self.seq_len, self.d_model))

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

        # Factory with kwargs
        model_mg = ToyTransformerBlock(d_model=self.d_model)
        comp_mg = modegpt_factory(model_mg, rank=32)
        out_mg = comp_mg(self.dummy_x)
        self.assertEqual(out_mg.shape, (self.batch_size, self.seq_len, self.d_model))

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

        # Factory with kwargs
        model_hf = ToyTransformerBlock(d_model=self.d_model)
        comp_hf = huffman_factory(model_hf, quantize_bits=4)
        out_hf = comp_hf(self.dummy_x)
        self.assertEqual(out_hf.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 12. UniQL Tests
    # ----------------------------------------------------------------------
    def test_uniql_compressor(self):
        """Test UniQL joint low-rank + uniform quantization compressor."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = UniQLCompressor(model, rank=16, bits=8)
        comp_model, stats = compressor.compress()

        self.assertEqual(stats["compressor_name"], "uniql")
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(out).any())

        # Factory with kwargs
        model_u = ToyTransformerBlock(d_model=self.d_model)
        comp_u = uniql_factory(model_u, rank=8, bits=4)
        out_u = comp_u(self.dummy_x)
        self.assertEqual(out_u.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 13. SDRM Tests
    # ----------------------------------------------------------------------
    def test_sdrm_compressor(self):
        """Test SDRM budget-proportional rank allocation compressor."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = SDRMCompressor(model, total_param_ratio=0.5)
        comp_model, stats = compressor.compress()

        self.assertEqual(stats["compressor_name"], "sdrm")
        self.assertIn("total_param_ratio", stats)
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(out).any())

        # Factory with kwargs
        model_s = ToyTransformerBlock(d_model=self.d_model)
        comp_s = sdrm_compressor_factory(model_s, total_param_ratio=0.6)
        out_s = comp_s(self.dummy_x)
        self.assertEqual(out_s.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 14. ENLighten Tests (Vectorized 2:4 Sparsity)
    # ----------------------------------------------------------------------
    def test_enlighten_compressor(self):
        """Test ENLighten low-rank SVD + 2:4 structured sparsity compressor."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = ENLightenCompressor(model, rank=16, sparsity=0.5)
        comp_model, stats = compressor.compress()

        self.assertEqual(stats["compressor_name"], "enlighten")
        self.assertIn("rank", stats)
        self.assertIn("sparsity", stats)
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(out).any())

        # Factory with kwargs
        model_e = ToyTransformerBlock(d_model=self.d_model)
        comp_e = enlighten_factory(model_e, rank=8, sparsity=0.5)
        out_e = comp_e(self.dummy_x)
        self.assertEqual(out_e.shape, (self.batch_size, self.seq_len, self.d_model))

    # ----------------------------------------------------------------------
    # 15. NLI Softmax Tests
    # ----------------------------------------------------------------------
    def test_nli_softmax_and_compressor(self):
        """Test NLISoftmax autograd function, backward stability, and model compressor."""
        # Test function forward and backward
        x = torch.randn(2, 4, 16, requires_grad=True)
        y = nli_softmax(x, dim=-1)
        self.assertEqual(y.shape, (2, 4, 16))
        self.assertTrue(torch.allclose(y.sum(dim=-1), torch.ones(2, 4)))

        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(x.grad).any())

        # Test NLISoftmax module
        mod = NLISoftmax(dim=-1)
        out_mod = mod(x.detach())
        self.assertEqual(out_mod.shape, (2, 4, 16))

        # Test NLISoftmaxCompressor
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = NLISoftmaxCompressor(model)
        comp_model, stats = compressor.compress()
        self.assertEqual(stats["compressor_name"], "nli_softmax")
        self.assertGreater(stats["softmax_modules_replaced"], 0)

        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(out).any())

        # Factory with kwargs
        model_nli = ToyTransformerBlock(d_model=self.d_model)
        comp_nli = nli_softmax_factory(model_nli)
        out_nli = comp_nli(self.dummy_x)
        self.assertEqual(out_nli.shape, (self.batch_size, self.seq_len, self.d_model))


if __name__ == "__main__":
    unittest.main()
