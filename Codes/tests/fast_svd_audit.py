"""
Fast Standalone Verification Suite for SVD and Low-Rank Specialist Audit.
Directly loads and tests the 11 targeted files without package-level glob side-effects:
1. attentions/svd_linear_attention.py
2. attentions/adaptive_svd_attention.py
3. attentions/asvd_attention.py
4. attentions/linformer_attention.py
5. compression_engine/res_svd.py
6. compression_engine/adaptive_ranks.py
7. compression_engine/direct_svd.py
8. compression_engine/dobi_svd.py
9. compression_engine/erc_svd.py
10. compression_engine/entropy_budget_allocation.py
11. compression_engine/kronecker_tt_lora.py
"""

import importlib.machinery
import importlib.util
import math
import os
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

codes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)

print("Starting Fast SVD Audit Runner...", flush=True)

# Register dummy package namespaces to avoid importing 200 unrelated modules
dummy_attn_pkg = types.ModuleType("framework.attentions")
dummy_attn_pkg.__path__ = [os.path.join(codes_dir, "framework/attentions")]
sys.modules["framework.attentions"] = dummy_attn_pkg

dummy_comp_pkg = types.ModuleType("framework.compression_engine")
dummy_comp_pkg.__path__ = [os.path.join(codes_dir, "framework/compression_engine")]
sys.modules["framework.compression_engine"] = dummy_comp_pkg

def load_source(module_name, rel_path, pkg_obj):
    abs_path = os.path.join(codes_dir, rel_path)
    loader = importlib.machinery.SourceFileLoader(module_name, abs_path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    short_name = module_name.split(".")[-1]
    setattr(pkg_obj, short_name, mod)
    loader.exec_module(mod)
    return mod

# Helper for base compressor imports that adaptive_ranks / direct_svd need
load_source("framework.compression_engine.aa_svd_compressor", "framework/compression_engine/aa_svd_compressor.py", dummy_comp_pkg)
load_source("framework.compression_engine.pruning_utils", "framework/compression_engine/pruning_utils.py", dummy_comp_pkg)
load_source("framework.attentions.compressed_attention", "framework/attentions/compressed_attention.py", dummy_attn_pkg)

# 1-4 Attentions
svd_lin_mod = load_source("framework.attentions.svd_linear_attention", "framework/attentions/svd_linear_attention.py", dummy_attn_pkg)
adap_attn_mod = load_source("framework.attentions.adaptive_svd_attention", "framework/attentions/adaptive_svd_attention.py", dummy_attn_pkg)
asvd_attn_mod = load_source("framework.attentions.asvd_attention", "framework/attentions/asvd_attention.py", dummy_attn_pkg)
linf_attn_mod = load_source("framework.attentions.linformer_attention", "framework/attentions/linformer_attention.py", dummy_attn_pkg)

# 5-11 Compression Engines
res_svd_mod = load_source("framework.compression_engine.res_svd", "framework/compression_engine/res_svd.py", dummy_comp_pkg)
adaptive_ranks_mod = load_source("framework.compression_engine.adaptive_ranks", "framework/compression_engine/adaptive_ranks.py", dummy_comp_pkg)
direct_svd_mod = load_source("framework.compression_engine.direct_svd", "framework/compression_engine/direct_svd.py", dummy_comp_pkg)
dobi_svd_mod = load_source("framework.compression_engine.dobi_svd", "framework/compression_engine/dobi_svd.py", dummy_comp_pkg)
erc_svd_mod = load_source("framework.compression_engine.erc_svd", "framework/compression_engine/erc_svd.py", dummy_comp_pkg)
entropy_budget_mod = load_source("framework.compression_engine.entropy_budget_allocation", "framework/compression_engine/entropy_budget_allocation.py", dummy_comp_pkg)
kronecker_tt_mod = load_source("framework.compression_engine.kronecker_tt_lora", "framework/compression_engine/kronecker_tt_lora.py", dummy_comp_pkg)

SVDLinear = svd_lin_mod.SVDLinear
SVDLinearAttention = svd_lin_mod.SVDLinearAttention
AdaptiveSVDAttention = adap_attn_mod.AdaptiveSVDAttention
AdaptiveSVDLinear = adap_attn_mod.AdaptiveSVDLinear
ASVDAttention = asvd_attn_mod.ASVDAttention
ASVDLinear = asvd_attn_mod.ASVDLinear
LinformerAttention = linf_attn_mod.LinformerAttention

ResSVDLinear = res_svd_mod.ResSVDLinear
ResSVDCompressor = res_svd_mod.ResSVDCompressor
SpectralResidualLoRA = res_svd_mod.SpectralResidualLoRA
AdaptiveRanksCompressor = adaptive_ranks_mod.AdaptiveRanksCompressor
DirectSVDCompressor = direct_svd_mod.DirectSVDCompressor
DobiSVDLinearLayer = dobi_svd_mod.DobiSVDLinearLayer
DobiSVDCompressor = dobi_svd_mod.DobiSVDCompressor
ERCCompressor = erc_svd_mod.ERCCompressor
EntropyBudgetCompressor = entropy_budget_mod.EntropyBudgetCompressor
KroneckerTTLoRALayer = kronecker_tt_mod.KroneckerTTLoRALayer
KroneckerTTLoRACompressor = kronecker_tt_mod.KroneckerTTLoRACompressor

print("All 11 modules successfully imported into clean sandbox!", flush=True)

def get_config():
    return types.SimpleNamespace(
        hidden_size=768,
        num_attention_heads=12,
        num_heads=12,
        dropout=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        intermediate_size=3072,
        rank=64,
        svd_rank=64,
        asvd_rank=64,
        residual_rank=8,
        energy_threshold=0.92,
        linear_attention_eps=1e-6,
    )

class MockMHA(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

class SimpleModel(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
    def forward(self, x):
        return self.classifier(self.fc2(F.relu(self.fc1(x))))

cfg = get_config()
x_768 = torch.randn(2, 16, 768, requires_grad=True)
mask_2d = torch.ones(2, 16)
mask_2d[:, 12:] = 0

print("\n--- 1. svd_linear_attention.py ---", flush=True)
svd_lin = SVDLinear(768, 768, rank=64, bias=True, freeze_svd=False, lora_rank=8)
out1 = svd_lin(x_768)
loss1 = out1.sum() + svd_lin.get_orthogonality_loss()
loss1.backward()
assert x_768.grad is not None and not torch.isnan(x_768.grad).any()
print("  [PASS] SVDLinear forward/backward/Stiefel", flush=True)

# exact mode
attn_exact = SVDLinearAttention(cfg, rank=64, mode="exact")
x_768.grad = None
out_ex, probs_ex = attn_exact(x_768, attention_mask=mask_2d, output_attentions=True)
out_ex.sum().backward()
assert x_768.grad is not None
print("  [PASS] SVDLinearAttention (exact mode)", flush=True)

# linear mode
attn_lin = SVDLinearAttention(cfg, rank=64, mode="linear", use_decay_gating=False)
x_768.grad = None
out_l, _ = attn_lin(x_768, attention_mask=mask_2d)
out_l.sum().backward()
assert x_768.grad is not None
print("  [PASS] SVDLinearAttention (linear kernel mode)", flush=True)

# decay mode
attn_dec = SVDLinearAttention(cfg, rank=64, mode="decay", use_decay_gating=True)
x_768.grad = None
out_d, _ = attn_dec(x_768, attention_mask=mask_2d)
out_d.sum().backward()
assert x_768.grad is not None
print("  [PASS] SVDLinearAttention (decay gated mode)", flush=True)

# step_recurrent
x_single = torch.randn(1, 4, 768)
out_seq, _, _ = attn_dec(x_single, return_state=True)
state = None
step_outs = []
for t in range(4):
    o_t, state = attn_dec.step_recurrent(x_single[:, t:t+1, :], past_state=state)
    step_outs.append(o_t)
diff = (out_seq - torch.cat(step_outs, dim=1)).abs().max().item()
assert diff < 1e-5
print(f"  [PASS] SVDLinearAttention (step_recurrent vs seq diff={diff:.2e})", flush=True)

print("\n--- 2. adaptive_svd_attention.py ---", flush=True)
adap_attn = AdaptiveSVDAttention(cfg, svd_rank=64, residual_rank=8, energy_threshold=0.92)
adap_attn.copy_weights(MockMHA(768))
x_768.grad = None
out_adap, _ = adap_attn(x_768, attention_mask=mask_2d)
out_adap.sum().backward()
assert x_768.grad is not None
S_test = torch.tensor([10.0, 5.0, 3.0, 1.0, 0.5, 0.2, 0.1, 0.05])
r_calc = adap_attn._compute_energy_rank(S_test, max_rank=8, min_rank=1)
assert r_calc == 2
print(f"  [PASS] AdaptiveSVDAttention forward/backward/energy_rank={r_calc}", flush=True)

print("\n--- 3. asvd_attention.py ---", flush=True)
asvd_attn = ASVDAttention(cfg, rank=64)
asvd_attn.copy_weights(MockMHA(768))
x_768.grad = None
out_asvd, _ = asvd_attn(x_768)
out_asvd.sum().backward()
assert x_768.grad is not None
print("  [PASS] ASVDAttention forward/backward", flush=True)

print("\n--- 4. linformer_attention.py ---", flush=True)
linf_attn = LinformerAttention(cfg, k=32, max_seq_len=128)
linf_attn.copy_weights(MockMHA(768))
x_768.grad = None
out_linf, _ = linf_attn(x_768, attention_mask=mask_2d)
out_linf.sum().backward()
assert x_768.grad is not None
print("  [PASS] LinformerAttention forward/backward", flush=True)

print("\n--- 5. res_svd.py ---", flush=True)
dense = nn.Linear(128, 128)
res_layer = ResSVDLinear.from_linear(dense, base_rank=32, residual_rank=8, scaling=1.0)
x_128 = torch.randn(2, 10, 128, requires_grad=True)
out_res = res_layer(x_128)
out_res.sum().backward()
assert x_128.grad is not None
assert res_layer.base_U.grad is None
assert res_layer.lora_A.grad is not None
err_norm = res_layer.get_residual_error_norm(dense.weight.data)
merged_lin = res_layer.merge_weights()
merge_diff = (res_layer(x_128.detach()) - merged_lin(x_128.detach())).abs().max().item()
assert merge_diff < 1e-4
print(f"  [PASS] ResSVDLinear forward/backward/tail_init (merge_diff={merge_diff:.2e}, err_norm={err_norm:.4f})", flush=True)

comp_res, stats_res = ResSVDCompressor(SimpleModel(128), base_rank=32, residual_rank=8).compress()
out_comp_res = comp_res(torch.randn(2, 128))
assert out_comp_res.shape == (2, 2)
print("  [PASS] ResSVDCompressor full model compression", flush=True)

print("\n--- 6. adaptive_ranks.py ---", flush=True)
comp_adap, stats_adap = AdaptiveRanksCompressor(SimpleModel(128), energy_threshold=0.92, min_rank=16).compress()
out_comp_adap = comp_adap(torch.randn(2, 128))
assert out_comp_adap.shape == (2, 2)
print("  [PASS] AdaptiveRanksCompressor full model compression", flush=True)

print("\n--- 7. direct_svd.py ---", flush=True)
comp_dir, stats_dir = DirectSVDCompressor(SimpleModel(128), rank=32, min_rank=16).compress()
out_comp_dir = comp_dir(torch.randn(2, 128))
assert out_comp_dir.shape == (2, 2)
print("  [PASS] DirectSVDCompressor full model compression", flush=True)

print("\n--- 8. dobi_svd.py ---", flush=True)
dobi_layer = DobiSVDCompressor(nn.Sequential(dense), rank=32).compress_linear(dense, rank=32)
x_128.grad = None
out_dobi = dobi_layer(x_128)
out_dobi.sum().backward()
assert x_128.grad is not None
assert dobi_layer.gate.grad is not None
comp_dobi, stats_dobi = DobiSVDCompressor(SimpleModel(128), rank=32).compress()
out_comp_dobi = comp_dobi(torch.randn(2, 128))
assert out_comp_dobi.shape == (2, 2)
print("  [PASS] DobiSVDLinearLayer & DobiSVDCompressor forward/backward/gating", flush=True)

print("\n--- 9. erc_svd.py ---", flush=True)
allowed_eps = 0.10
erc_comp = ERCCompressor(SimpleModel(128), allowed_error=allowed_eps)
erc_res = erc_comp.compress_linear(dense)
erc_svd_layer = erc_res[0] if isinstance(erc_res, tuple) else erc_res
W_orig = dense.weight.data.float()
W_app = (erc_svd_layer.U @ erc_svd_layer.V).float()
frob_err = (torch.norm(W_orig - W_app, p='fro') / torch.norm(W_orig, p='fro')).item()
assert frob_err <= allowed_eps + 1e-4
comp_erc, stats_erc = erc_comp.compress()
out_comp_erc = comp_erc(torch.randn(2, 128))
assert out_comp_erc.shape == (2, 2)
print(f"  [PASS] ERCCompressor Frobenius error bound verified ({frob_err:.4f} <= {allowed_eps})", flush=True)

print("\n--- 10. entropy_budget_allocation.py ---", flush=True)
ent_comp = EntropyBudgetCompressor(SimpleModel(128), total_budget_ratio=0.7)
comp_ent, stats_ent = ent_comp.compress()
for name, r in stats_ent["allocated_ranks"].items():
    assert r >= 16
out_comp_ent = comp_ent(torch.randn(2, 128))
assert out_comp_ent.shape == (2, 2)
print("  [PASS] EntropyBudgetCompressor rank allocation (all ranks >= 16)", flush=True)

print("\n--- 11. kronecker_tt_lora.py ---", flush=True)
kron_layer = KroneckerTTLoRALayer.from_linear(dense, tt_ranks=(1, 4, 4, 1), gamma=1.0, lora_r=4, lora_alpha=8.0)
x_128.grad = None
out_kron = kron_layer(x_128)
out_kron.sum().backward()
assert x_128.grad is not None
assert kron_layer.G1.grad is not None
assert kron_layer.lora_A.grad is not None
kron_layer.adapt_tt_ranks(delta_r=1)
out_kron_after = kron_layer(x_128)
assert out_kron_after.shape == (2, 10, 128)
print("  [PASS] KroneckerTTLoRALayer forward/backward/adapt_tt_ranks", flush=True)

print("\n=======================================================", flush=True)
print(">>> ALL 11 SVD / FACTORIZATION MODULES VERIFIED 100% SUCCESSFUL <<<", flush=True)
print("=======================================================", flush=True)
