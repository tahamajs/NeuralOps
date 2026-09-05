#!/usr/bin/env python3
import sys
import os
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn

sys.path.insert(0, 'Codes')
sys.path.insert(0, '.')

from framework.core.registry import COMPRESSOR_REGISTRY
import framework.compression_engine
from framework.utils.compat import should_compress_linear, get_dynamic_compression_rank

# ==============================================================================
# ARCHITECTURE MODELS
# ==============================================================================

class DistilBertBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.q_lin = nn.Linear(d, d)
        self.k_lin = nn.Linear(d, d)
        self.v_lin = nn.Linear(d, d)
        self.out_lin = nn.Linear(d, d)
        self.lin1 = nn.Linear(d, d * 4)
        self.lin2 = nn.Linear(d * 4, d)

    def forward(self, x):
        q = self.q_lin(x)
        k = self.k_lin(x)
        v = self.v_lin(x)
        ctx = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1]**0.5), dim=-1), v)
        out = self.out_lin(ctx) + x
        return self.lin2(torch.relu(self.lin1(out))) + out

class DistilBertModel(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.config = SimpleNamespace(model_type="distilbert", hidden_size=d)
        self.layer = nn.ModuleList([DistilBertBlock(d) for _ in range(2)])
        self.classifier = nn.Linear(d, 2)

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return self.classifier(x.mean(dim=1))


class RobertaSelfAttention(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1]**0.5), dim=-1), v)

class RobertaAttention(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.self = RobertaSelfAttention(d)
        self.output = nn.Module()
        self.output.dense = nn.Linear(d, d)

    def forward(self, x):
        return self.output.dense(self.self(x)) + x

class RobertaBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.attention = RobertaAttention(d)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(d, d * 4)
        self.output = nn.Module()
        self.output.dense = nn.Linear(d * 4, d)

    def forward(self, x):
        h = self.attention(x)
        return self.output.dense(torch.relu(self.intermediate.dense(h))) + h

class RobertaModel(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.config = SimpleNamespace(model_type="roberta", hidden_size=d)
        self.layer = nn.ModuleList([RobertaBlock(d) for _ in range(2)])
        self.classifier = nn.Linear(d, 2)

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return self.classifier(x.mean(dim=1))


class GPT2Block(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.c_attn = nn.Linear(d, d * 3)
        self.c_proj = nn.Linear(d, d)
        self.c_fc = nn.Linear(d, d * 4)
        self.mlp_c_proj = nn.Linear(d * 4, d)

    def forward(self, x):
        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        ctx = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1]**0.5), dim=-1), v)
        out = self.c_proj(ctx) + x
        return self.mlp_c_proj(torch.relu(self.c_fc(out))) + out

class GPT2Model(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.config = SimpleNamespace(model_type="gpt2", hidden_size=d, is_decoder=True)
        self.h = nn.ModuleList([GPT2Block(d) for _ in range(2)])
        self.lm_head = nn.Linear(d, 2)

    def forward(self, x):
        for b in self.h:
            x = b(x)
        return self.lm_head(x.mean(dim=1))


class Qwen35Block(nn.Module):
    def __init__(self, d=64, ffn_dim=128):
        super().__init__()
        self.delta_net = nn.Module()
        self.delta_net.in_proj_qkv = nn.Linear(d, d * 3)
        self.delta_net.out_proj = nn.Linear(d, d)

        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(d, d)
        self.attn.k_proj = nn.Linear(d, 16)
        self.attn.v_proj = nn.Linear(d, 16)
        self.attn.o_proj = nn.Linear(d, d)

        self.expert_router = nn.Linear(d, 4)

        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(d, ffn_dim)
        self.mlp.up_proj = nn.Linear(d, ffn_dim)
        self.mlp.down_proj = nn.Linear(ffn_dim, d)

    def forward(self, x):
        h = self.mlp.down_proj(torch.relu(self.mlp.up_proj(x)))
        return x + h

class Qwen35Model(nn.Module):
    def __init__(self, d=64, ffn_dim=128):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_5",
            hidden_size=d,
            num_attention_heads=4,
            num_key_value_heads=1,
            num_hidden_layers=2,
            intermediate_size=ffn_dim,
            is_decoder=True,
        )
        self.layers = nn.ModuleList([Qwen35Block(d, ffn_dim) for _ in range(2)])
        self.score = nn.Linear(d, 2)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.score(x.mean(dim=1))


# ==============================================================================
# COMPRESSOR RUNNER
# ==============================================================================

def execute_compressor(cls, model):
    strategies = [
        lambda: cls(model, rank=16),
        lambda: cls(model, target_rank=16),
        lambda: cls(model),
        lambda: cls(model, sparsity=0.5),
        lambda: cls(model, bits=4),
        lambda: cls(model, rank=8),
    ]
    compressor = None
    last_err = ""
    for s in strategies:
        try:
            compressor = s()
            break
        except Exception as e:
            last_err = str(e)

    if compressor is None:
        raise RuntimeError(f"Instantiation failed: {last_err}")

    if hasattr(compressor, "compress"):
        res = compressor.compress()
        compressed_model = res[0] if isinstance(res, tuple) else res
    elif callable(compressor) and not isinstance(compressor, nn.Module):
        res = compressor()
        compressed_model = res[0] if isinstance(res, tuple) else res
    elif isinstance(compressor, nn.Module):
        compressed_model = compressor
    else:
        compressed_model = getattr(compressor, "model", model)

    if compressed_model is None or not isinstance(compressed_model, nn.Module):
        compressed_model = model

    return compressed_model


def run_multiarch_tests():
    names = sorted([k for k in COMPRESSOR_REGISTRY.keys() if k not in ("your_compressor_name", "replaceme")])
    print("=" * 80)
    print(f"AUDITING {len(names)} COMPRESSORS ACROSS 4 ARCHITECTURES")
    print("=" * 80)

    total_passed = 0
    total_tests = 0
    arch_failures = {"distilbert": [], "roberta": [], "gpt2": [], "qwen3_5": []}

    for idx, name in enumerate(names, 1):
        cls = COMPRESSOR_REGISTRY.resolve(name)
        print(f"[{idx:3d}/{len(names):3d}] >> {name}")

        # 1. DistilBERT
        try:
            m_db = DistilBertModel(64)
            c_db = execute_compressor(cls, m_db)
            x = torch.randn(2, 8, 64)
            out = c_db(x)
            if isinstance(out, tuple): out = out[0]
            assert out.shape == (2, 2) and torch.isfinite(out).all()
            print("   [PASS] DistilBERT")
            total_passed += 1
        except Exception as e:
            print(f"   [FAIL] DistilBERT: {e}")
            arch_failures["distilbert"].append((name, str(e)))
        total_tests += 1

        # 2. RoBERTa
        try:
            m_rob = RobertaModel(64)
            c_rob = execute_compressor(cls, m_rob)
            x = torch.randn(2, 8, 64)
            out = c_rob(x)
            if isinstance(out, tuple): out = out[0]
            assert out.shape == (2, 2) and torch.isfinite(out).all()
            print("   [PASS] RoBERTa")
            total_passed += 1
        except Exception as e:
            print(f"   [FAIL] RoBERTa: {e}")
            arch_failures["roberta"].append((name, str(e)))
        total_tests += 1

        # 3. GPT-2
        try:
            m_gpt = GPT2Model(64)
            c_gpt = execute_compressor(cls, m_gpt)
            x = torch.randn(2, 8, 64)
            out = c_gpt(x)
            if isinstance(out, tuple): out = out[0]
            assert out.shape == (2, 2) and torch.isfinite(out).all()
            print("   [PASS] GPT-2")
            total_passed += 1
        except Exception as e:
            print(f"   [FAIL] GPT-2: {e}")
            arch_failures["gpt2"].append((name, str(e)))
        total_tests += 1

        # 4. Qwen 3.5
        try:
            m_qwen = Qwen35Model(64, 128)
            c_qwen = execute_compressor(cls, m_qwen)
            x = torch.randn(2, 8, 64)
            out = c_qwen(x)
            if isinstance(out, tuple): out = out[0]
            assert out.shape == (2, 2) and torch.isfinite(out).all()
            print("   [PASS] Qwen 3.5")
            total_passed += 1
        except Exception as e:
            print(f"   [FAIL] Qwen 3.5: {e}")
            arch_failures["qwen3_5"].append((name, str(e)))
        total_tests += 1

    print("\n" + "=" * 80)
    print(f"COMPRESSOR MULTI-ARCH SUMMARY: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
    print("=" * 80)
    for arch, fails in arch_failures.items():
        print(f"Architecture '{arch}': {len(names) - len(fails)}/{len(names)} passed ({len(fails)} failures)")
        if fails:
            for n, err in fails[:5]:
                print(f"   - {n}: {err}")

    return total_passed == total_tests

if __name__ == '__main__':
    success = run_multiarch_tests()
    sys.exit(0 if success else 1)
