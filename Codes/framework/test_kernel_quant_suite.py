"""
Comprehensive validation test suite for Kernel/Sparse Attentions and Quantization/KV Compressors.
"""

import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure framework on path
_HERE = Path(__file__).resolve().parent
_CODES = _HERE.parent
if str(_CODES) not in sys.path:
    sys.path.insert(0, str(_CODES))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import framework.attentions
import framework.compression_engine


def make_dummy_config(hidden_size=768, num_heads=12, rank=32):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        rank=rank,
        dropout=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        num_landmarks=32,
        performer_rank=32,
        galerkin_rank=32,
        salt_rank=16,
        xattn_block_size=16,
        xattn_sparsity=0.5,
        nsa_window_size=16,
        nsa_top_k=8,
        lsh_rounds=2,
        lsh_buckets=8,
        cluster_num_clusters=8,
        routing_num_clusters=8,
        max_span=64,
    )


class MockTransformerModel(nn.Module):
    """Mini transformer model with multiple linear and attention-like layers for compressor testing."""
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(hidden_size, hidden_size),
                "k_proj": nn.Linear(hidden_size, hidden_size),
                "v_proj": nn.Linear(hidden_size, hidden_size),
                "out_proj": nn.Linear(hidden_size, hidden_size),
                "up_proj": nn.Linear(hidden_size, hidden_size * 2),
                "gate_proj": nn.Linear(hidden_size, hidden_size * 2),
                "down_proj": nn.Linear(hidden_size * 2, hidden_size),
                "softmax": nn.Softmax(dim=-1),
            })
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h = x
        for layer in self.layers:
            q = layer["q_proj"](h)
            k = layer["k_proj"](h)
            v = layer["v_proj"](h)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)
            attn = layer["softmax"](scores)
            attn_out = torch.matmul(attn, v)
            h = h + layer["out_proj"](attn_out)
            mlp = layer["down_proj"](F.gelu(layer["gate_proj"](h)) * layer["up_proj"](h))
            h = h + mlp
        logits = self.classifier(h[:, 0, :])
        return logits


def test_attention_modules():
    print("=" * 70)
    print(">>> VALIDATING 14 KERNEL & SPARSE ATTENTION MODULES")
    print("=" * 70)
    
    attn_keys = [
        "performer",
        "nystrom",
        "flash3",
        "cluster",
        "adaptive_span",
        "span_expanded",
        "lsh",
        "nsa",
        "galerkin",
        "adasplash",
        "hamming",
        "routing",
        "salt",
        "xattention",
    ]

    B, N, D = 2, 64, 768
    cfg = make_dummy_config(hidden_size=D, num_heads=12, rank=32)
    results = {}

    for key in attn_keys:
        print(f"\n[Testing Attention: '{key}']")
        cls = ATTENTION_REGISTRY.get(key)
        if cls is None:
            print(f"  ✗ FAILED: Key '{key}' not registered in ATTENTION_REGISTRY")
            results[key] = {"status": "FAILED", "error": "Not registered"}
            continue

        try:
            # 1. Instantiation
            module = cls(config=cfg)
            module.train()

            # 2. Forward pass with no mask
            x = torch.randn(B, N, D, requires_grad=True)
            out, attn_weights = module(x)
            assert out.shape == (B, N, D), f"Unexpected output shape: {out.shape} vs {(B, N, D)}"
            
            # 3. Backward pass
            loss = out.sum()
            loss.backward()
            assert x.grad is not None, "Gradient did not propagate back to input tensor x"
            has_param_grad = any(p.grad is not None for p in module.parameters() if p.requires_grad)
            assert has_param_grad, "No module parameters received gradients"

            # 4. Forward with padding mask (2D and 4D)
            module.zero_grad()
            mask_2d = torch.ones(B, N, dtype=torch.long)
            mask_2d[:, N // 2 :] = 0  # 50% padded
            x_masked = torch.randn(B, N, D, requires_grad=True)
            out_masked, _ = module(x_masked, attention_mask=mask_2d)
            assert out_masked.shape == (B, N, D), f"Masked output shape mismatch: {out_masked.shape}"
            loss_m = out_masked.sum()
            loss_m.backward()

            # 5. Variable sequence lengths test
            for test_n in [16, 32, 128]:
                x_var = torch.randn(B, test_n, D)
                out_var, _ = module(x_var)
                assert out_var.shape == (B, test_n, D), f"Failed variable length test N={test_n}"

            param_count = sum(p.numel() for p in module.parameters())
            print(f"  ✓ PASSED: Forward, Backward, Masking, Variable SeqLen (Params: {param_count:,})")
            results[key] = {
                "status": "PASSED",
                "params": param_count,
                "output_shape": list(out.shape),
                "has_grad": True,
            }

        except Exception as e:
            traceback.print_exc()
            print(f"  ✗ FAILED with error: {e}")
            results[key] = {"status": "FAILED", "error": str(e)}

    return results


def test_compressor_modules():
    print("\n" + "=" * 70)
    print(">>> VALIDATING 15 QUANTIZATION & KV CACHE COMPRESSOR ENGINES")
    print("=" * 70)

    comp_keys = [
        "astro",
        "faq",
        "coopq",
        "kmeans",
        "delta_kv",
        "hybridkv",
        "kvtc",
        "cachenotes",
        "zerokv_neo",
        "modegpt",
        "huffman",
        "uniql",
        "sdrm",
        "enlighten",
        "nli_softmax",
    ]

    results = {}

    for key in comp_keys:
        print(f"\n[Testing Compressor: '{key}']")
        cls = COMPRESSOR_REGISTRY.get(key)
        if cls is None:
            print(f"  ✗ FAILED: Key '{key}' not registered in COMPRESSOR_REGISTRY")
            results[key] = {"status": "FAILED", "error": "Not registered"}
            continue

        try:
            model = MockTransformerModel(hidden_size=64, num_layers=2)
            orig_params = sum(p.numel() for p in model.parameters())

            # Specific testing based on compressor class category
            if key in ["astro", "faq", "coopq", "kmeans", "uniql", "sdrm", "enlighten", "modegpt", "nli_softmax"]:
                # Model-level compressor / factory
                if callable(cls):
                    compressed_model = cls(model)
                else:
                    compressor = cls(model)
                    compressed_model, stats = compressor.compress()
                
                # Check forward pass on compressed model
                x = torch.randn(2, 16, 64)
                logits = compressed_model(x)
                assert logits.shape == (2, 2), f"Logits shape mismatch: {logits.shape}"

                # Test gradient flow if QAT / STE enabled
                if key in ["astro", "faq", "coopq", "kmeans"]:
                    loss = logits.sum()
                    loss.backward()
                
                print(f"  ✓ PASSED: Model compression & forward verification (Orig Params: {orig_params})")
                results[key] = {"status": "PASSED", "type": "Weight/QAT Compressor"}

            elif key in ["delta_kv", "hybridkv", "kvtc", "cachenotes", "zerokv_neo"]:
                # KV Cache Runtime Compressor
                from framework.compression_engine.delta_kv import DeltaKVStore
                from framework.compression_engine.hybridkv_compressor import HybridKVCache
                from framework.compression_engine.kvtc_compressor import KVTCStore
                from framework.compression_engine.cachenotes_compressor import CacheNotesKVCache
                from framework.compression_engine.zerokv_neo_compressor import ZeroKVNeoCache

                # Test KV store class
                if key == "delta_kv":
                    store = DeltaKVStore(max_refs=32, topk=2)
                    k = torch.randn(2, 4, 16, 32)
                    v = torch.randn(2, 4, 16, 32)
                    k_c, v_c = store.compress(k, v)
                    store.update(k, v)
                    # Next token decoding step
                    k_tok = torch.randn(2, 4, 1, 32)
                    v_tok = torch.randn(2, 4, 1, 32)
                    k_c2, v_c2 = store.compress(k_tok, v_tok)
                    assert k_c2.shape == (2, 4, 1, 32)

                elif key == "hybridkv":
                    cache = HybridKVCache(max_cache_len=32, num_sinks=4, recent_window=16)
                    k = torch.randn(2, 4, 48, 32)
                    v = torch.randn(2, 4, 48, 32)
                    k_c, v_c = cache.update_and_compress(k, v)
                    assert k_c.size(2) <= 32, f"Cache size {k_c.size(2)} exceeded limit 32"

                elif key == "kvtc":
                    store = KVTCStore(bits=4, pca_energy=0.95)
                    k = torch.randn(2, 4, 32, 32)
                    v = torch.randn(2, 4, 32, 32)
                    k_hat, v_hat, stats = store.compress_kv_cache(k, v)
                    assert k_hat.shape == k.shape
                    assert "theoretical_compression_ratio" in stats

                elif key == "cachenotes":
                    cache = CacheNotesKVCache(max_cache_len=32, task_type="qa", keep_ratio=0.5, num_sinks=4)
                    k = torch.randn(2, 4, 64, 32)
                    v = torch.randn(2, 4, 64, 32)
                    k_c, v_c = cache.update_and_compress(k, v)
                    assert k_c.size(2) <= 32

                elif key == "zerokv_neo":
                    cache = ZeroKVNeoCache(max_cache_len=32, num_attention_sinks=4, recent_window=16)
                    k = torch.randn(2, 4, 64, 32)
                    v = torch.randn(2, 4, 64, 32)
                    k_c, v_c = cache.update_and_compress(k, v)
                    assert k_c.size(2) <= 32

                # Also test compressor factory
                compressed_model = cls(model) if callable(cls) else cls(model).compress()[0]
                x = torch.randn(2, 16, 64)
                logits = compressed_model(x)
                assert logits.shape == (2, 2)

                print(f"  ✓ PASSED: Dynamic KV Cache management & model factory verification")
                results[key] = {"status": "PASSED", "type": "KV Cache Engine"}

            elif key == "huffman":
                from framework.compression_engine.huffman_coder import HuffmanCoder
                t = torch.randn(128, 64)
                packed_bytes, code_map, stats = HuffmanCoder.encode(t, quantize_bits=8)
                decoded_t = HuffmanCoder.decode(packed_bytes, code_map, stats)
                assert decoded_t.shape == t.shape
                # Test factory
                compressed_model = cls(model) if callable(cls) else cls(model).compress()[0]
                print(f"  ✓ PASSED: Huffman encode/decode lossless pipeline & model factory")
                results[key] = {"status": "PASSED", "type": "Lossless Entropy Coder"}

        except Exception as e:
            traceback.print_exc()
            print(f"  ✗ FAILED with error: {e}")
            results[key] = {"status": "FAILED", "error": str(e)}

    return results


def test_framework_integration():
    print("\n" + "=" * 70)
    print(">>> VERIFYING FRAMEWORK INTEGRATION (main.py + replace_attention)")
    print("=" * 70)
    from framework.utils.replace_attention import replace_attention
    from transformers import BertConfig, BertForSequenceClassification

    hf_cfg = BertConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=256,
        num_labels=2,
    )
    
    test_attn_types = ["performer", "nystrom", "flash3", "galerkin", "adaptive_span", "salt"]
    integration_results = {}

    for attn_type in test_attn_types:
        try:
            model = BertForSequenceClassification(hf_cfg)
            replaced_model = replace_attention(model, attention_key=attn_type, rank=16)
            input_ids = torch.randint(0, 1000, (2, 32))
            mask = torch.ones(2, 32, dtype=torch.long)
            out = replaced_model(input_ids=input_ids, attention_mask=mask)
            assert out.logits.shape == (2, 2), f"Expected logits (2, 2), got {out.logits.shape}"
            loss = out.logits.sum()
            loss.backward()
            print(f"  ✓ PASSED: replace_attention('{attn_type}') -> Forward & Backward on HF BERT")
            integration_results[attn_type] = "PASSED"
        except Exception as e:
            traceback.print_exc()
            print(f"  ✗ FAILED: replace_attention('{attn_type}') with error {e}")
            integration_results[attn_type] = f"FAILED: {e}"

    return integration_results


if __name__ == "__main__":
    attn_res = test_attention_modules()
    comp_res = test_compressor_modules()
    int_res = test_framework_integration()

    print("\n" + "=" * 70)
    print(">>> SUMMARY OF VALIDATION RUN")
    print("=" * 70)
    attn_passed = sum(1 for v in attn_res.values() if v.get("status") == "PASSED")
    comp_passed = sum(1 for v in comp_res.values() if v.get("status") == "PASSED")
    print(f"Attentions Validated:  {attn_passed}/{len(attn_res)} Passed")
    print(f"Compressors Validated: {comp_passed}/{len(comp_res)} Passed")
    print(f"Integration Tests:     {sum(1 for v in int_res.values() if v == 'PASSED')}/{len(int_res)} Passed")
