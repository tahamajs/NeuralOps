"""
Comprehensive Test Suite for Vectorized Huffman Coder & Huffman Compressor.
Tests roundtrip correctness, compression metrics, fallback LoRA freezing,
and cross-architecture integration.
"""
import math
import time
import unittest

import torch
import torch.nn as nn
from framework.compression_engine.huffman_coder import (HuffmanCoder,
                                                        HuffmanCompressor,
                                                        _freeze_base_only)


class DummyLoRAModule(nn.Module):
    def __init__(self, in_features=64, out_features=64, r=16):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))


class DummyTransformerModel(nn.Module):
    def __init__(self, d_model=64, num_layers=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(100, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(d_model, d_model),
                "k_proj": nn.Linear(d_model, d_model),
                "v_proj": nn.Linear(d_model, d_model),
                "out_proj": nn.Linear(d_model, d_model),
            })
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = torch.zeros((1, 4), dtype=torch.long)
        x = self.word_embeddings(input_ids)
        for layer in self.layers:
            q = layer["q_proj"](x)
            k = layer["k_proj"](x)
            v = layer["v_proj"](x)
            x = x + layer["out_proj"](v)
        return self.classifier(x[:, 0])


class TestHuffmanCoder(unittest.TestCase):

    # 2.1 Roundtrip Correctness
    def test_roundtrip_small_tensor(self):
        x = torch.randn(1000)
        packed, table, meta = HuffmanCoder.encode(x, quantize_bits=8)
        self.assertGreater(len(packed), 0)
        rec = HuffmanCoder.decode(packed, table, meta)
        err = torch.norm(x - rec) / torch.norm(x)
        self.assertLess(err.item(), 0.05, f"Small tensor error {err.item():.4f} too high")

    def test_roundtrip_large_tensor(self):
        x = torch.randn(1, 512, 768)
        t0 = time.time()
        packed, table, meta = HuffmanCoder.encode(x, quantize_bits=8)
        rec = HuffmanCoder.decode(packed, table, meta)
        elapsed = time.time() - t0
        err = torch.norm(x - rec) / torch.norm(x)
        self.assertLess(elapsed, 15.0, f"Encode+decode took {elapsed:.2f}s, expected < 15s")
        self.assertLess(err.item(), 0.05)

    def test_roundtrip_zero_tensor(self):
        x = torch.zeros(100, 100)
        packed, table, meta = HuffmanCoder.encode(x, quantize_bits=8)
        rec = HuffmanCoder.decode(packed, table, meta)
        self.assertEqual(rec.shape, x.shape)
        self.assertTrue(torch.allclose(x, rec, atol=1e-5))

    def test_roundtrip_high_bits(self):
        x = torch.randn(128, 128)
        packed, table, meta = HuffmanCoder.encode(x, quantize_bits=16)
        rec = HuffmanCoder.decode(packed, table, meta)
        self.assertTrue(torch.allclose(x, rec, atol=1e-5), "High-bit quantization must be lossless identity")
        self.assertTrue(meta.get("is_raw", False))

    # 2.2 Compression Metrics
    def test_entropy_calculation(self):
        # 4 symbols with uniform frequency: entropy should be log2(4) = 2.0
        x = torch.tensor([0.0, 1.0, 2.0, 3.0] * 250)
        packed, table, meta = HuffmanCoder.encode(x, quantize_bits=2)
        entropy = meta["entropy"]
        self.assertAlmostEqual(entropy, 2.0, places=1)

    def test_compression_ratio(self):
        # Skewed tensor: mostly zeros
        x = torch.zeros(1000)
        x[0:10] = 5.0
        packed, table, meta = HuffmanCoder.encode(x, quantize_bits=8)
        # Original bytes: 1000 * 4 = 4000
        # Compressed should be much smaller
        orig_bytes = x.numel() * x.element_size()
        self.assertLess(meta["compressed_bytes"], orig_bytes)

    # 2.3 Fallback and LoRA Freezing
    def test_fallback_freezes_only_base(self):
        mod = DummyLoRAModule(64, 64, r=16)
        _freeze_base_only(mod)
        self.assertTrue(mod.lora_A.requires_grad, "lora_A should stay trainable")
        self.assertTrue(mod.lora_B.requires_grad, "lora_B should stay trainable")
        self.assertFalse(mod.base_layer.weight.requires_grad, "base_layer.weight should be frozen")
        if mod.base_layer.bias is not None:
            self.assertFalse(mod.base_layer.bias.requires_grad, "base_layer.bias should be frozen")

    def test_non_lora_fallback_freezes_all(self):
        mod = nn.Linear(64, 64)
        _freeze_base_only(mod)
        self.assertFalse(mod.weight.requires_grad, "Plain linear weight should be frozen")
        self.assertFalse(mod.bias.requires_grad, "Plain linear bias should be frozen")

    # 2.4 Cross-Architecture Integration
    def test_injection_transformer(self):
        model = DummyTransformerModel(d_model=64, num_layers=2)
        comp = HuffmanCompressor(model, quantize_bits=8, max_error=0.20)
        c_model, stats = comp.compress()
        self.assertGreater(stats["num_compressed"], 0)
        self.assertGreater(stats["huffman_compression_ratio"], 1.0)
        self.assertTrue(stats.get("forward_check_success"))


if __name__ == "__main__":
    unittest.main()
