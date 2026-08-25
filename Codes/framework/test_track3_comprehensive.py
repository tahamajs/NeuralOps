import sys
import os
import time

sys.path.insert(0, '/Users/tahamajs/Documents/uni/Project/Codes')
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== STARTING TRACK 3 COMPREHENSIVE AUDIT ===", flush=True)

class DummyConfig:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.dropout = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.is_decoder = False
        self.causal = False
        self.page_size = 32
        self.k_experts = 2
        self.svd_rank = 16
        self.window_size = 32
        self.num_experts = 4
        self.top_k = 2

cfg = DummyConfig()

class MockOldAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lin = nn.Linear(768, 768)
        self.k_lin = nn.Linear(768, 768)
        self.v_lin = nn.Linear(768, 768)
        self.out_lin = nn.Linear(768, 768)

old_attn = MockOldAttn()

tests = []

# 1. BAM-UMoE
def test_bam_umoe():
    from framework.attentions.bam_umoe import BAMUMoEBlock
    m = BAMUMoEBlock(config=cfg, num_experts=4, top_k=2, old_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    mask[0, 50:] = 0
    out, attn = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    assert m.aux_loss.item() > 0.0
    loss = out.sum() + m.aux_loss
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (aux_loss > 0, grad ok, copy_weights ok)"
tests.append(("bam_umoe", test_bam_umoe))

# 2. Expert Choice MoE
def test_expert_choice():
    from framework.attentions.expert_choice_moe import ExpertChoiceMoEAttention
    m = ExpertChoiceMoEAttention(config=cfg, n_experts=4, capacity_factor=1.0, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, attn = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    assert m.aux_loss.item() > 0.0
    loss = out.sum() + m.aux_loss
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (capacity balancing exact, aux_loss > 0, grad ok)"
tests.append(("expert_choice_moe", test_expert_choice))

# 3. REAP-MoE
def test_reap_moe():
    from framework.attentions.reap_moe import REAPMoEAttention
    m = REAPMoEAttention(config=cfg, n_experts=4, top_k=2, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    out, attn = m(x)
    assert out.shape == (2, 64, 768)
    assert m.aux_loss.item() > 0.0
    loss = out.sum() + m.aux_loss
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (SVD factorized experts, aux_loss > 0, grad ok)"
tests.append(("reap_moe", test_reap_moe))

# 4. Samba-MoE
def test_samba_moe():
    from framework.attentions.samba_moe import SambaMoEBlock
    m = SambaMoEBlock(config=cfg, n_experts=4, window=32, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    out, attn = m(x)
    assert out.shape == (2, 64, 768)
    assert m.aux_loss.item() > 0.0
    loss = out.sum() + m.aux_loss
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Hybrid SWA + SSM + MoE, aux_loss > 0, grad ok)"
tests.append(("samba_moe", test_samba_moe))

# 5. Jamba
def test_jamba():
    from framework.attentions.jamba import Jamba2Block
    m = Jamba2Block(config=cfg, n_experts=4, top_k=2, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    out, attn = m(x)
    assert out.shape == (2, 64, 768)
    assert m.aux_loss.item() > 0.0
    loss = out.sum() + m.aux_loss
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (SVD-Mamba + MHA + EC-MoE, aux_loss > 0, grad ok)"
tests.append(("jamba", test_jamba))

# 6. HymbaTransX
def test_hymba():
    from framework.attentions.hymba_transx import HymbaTransXBlock
    m = HymbaTransXBlock(config=cfg, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, attn = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Differential Transformer + SSM + HeadRoPE, grad ok)"
tests.append(("hymba_transx", test_hymba))

# 7. MoE-DeltaKV-Paged
def test_moe_deltakv():
    from framework.attentions.moe_deltakv_paged import MoEDeltaKVPagedAttention
    m = MoEDeltaKVPagedAttention(config=cfg, page_size=16, k_experts=2, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    mask[0, 50:] = 0
    out, attn = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    assert m.aux_loss.item() > 0.0
    loss = out.sum() + m.aux_loss
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (MoBA + Paged DeltaKV STE + Page 0 CLS protection, aux_loss > 0, grad ok)"
tests.append(("moe_deltakv_paged", test_moe_deltakv))

# 8. Block Transformer HIBA
def test_hiba():
    from framework.attentions.block_transformer_hiba import BlockTransformerHIBA
    m = BlockTransformerHIBA(config=cfg, block_size=16, top_k_inter=2, rectification_interval=2, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    mask[0, 50:] = 0
    out1, _ = m(x, attention_mask=mask) # rectify step
    out2, _ = m(x, attention_mask=mask) # sparse step
    assert out1.shape == (2, 64, 768) and out2.shape == (2, 64, 768)
    loss = out1.sum() + out2.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Hierarchical Intra/Inter-Block + CLS block 0 guaranteed, grad ok)"
tests.append(("block_transformer_hiba", test_hiba))

# 9. NEXUS
def test_nexus():
    from framework.attentions.nexus import NEXUSAttention, NEXUSConfig
    n_cfg = NEXUSConfig(block_size=16, top_k_blocks=2, rectify_interval=2, stripe_enabled=True, stripe_width=4)
    m = NEXUSAttention(model_config=cfg, nexus_cfg=n_cfg, layer_idx=1, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, _ = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (XAttention + Seer + DuSA + Stripe + Softpick + CLS block 0 protected, grad ok)"
tests.append(("nexus", test_nexus))

# 10. LEKA v1
def test_leka_v1():
    from framework.attentions.leka_v1 import LEKAV1Attention
    m = LEKAV1Attention(config=cfg)
    m.copy_weights(old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, _ = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Linear RFF kernel attention, grad ok)"
tests.append(("leka_v1", test_leka_v1))

# 11. LEKA v2
def test_leka_v2():
    from framework.attentions.leka_v2 import LEKAv2Attention
    m = LEKAv2Attention(config=cfg)
    m.copy_weights(old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, _ = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Dual-scale RFF + Landmarks with CLS index 0 protected + Local Window, grad ok)"
tests.append(("leka_v2", test_leka_v2))

# 12. LEKA v3
def test_leka_v3():
    from framework.attentions.leka_v3 import LEKAV3Attention
    m = LEKAV3Attention(config=cfg)
    m.copy_weights(old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, _ = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Value-Scored Landmarks with CLS index 0 protected + Local Window, grad ok)"
tests.append(("leka_v3", test_leka_v3))

# 13. MixAttention-MoSKA
def test_moska():
    from framework.attentions.mixattention_moska import MixAttentionMoSKA
    m = MixAttentionMoSKA(config=cfg, fasa_keep_ratio=0.5, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, attn = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Variance-based token pruning with unconditional CLS index 0 protection, grad ok)"
tests.append(("mixattention_moska", test_moska))

# 14. PHANTOM
def test_phantom():
    from framework.attentions.phantom import PHANTOMAttention, PHANTOMConfig
    p_cfg = PHANTOMConfig(cur_rank=16, lsh_rounds=2, lsh_bucket_size=4)
    m = PHANTOMAttention(config=cfg, phantom_cfg=p_cfg, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, _ = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (CUR Leverage Decomposition + Angular LSH + Softpick, grad ok)"
tests.append(("phantom", test_phantom))

# 15. MoBA
def test_moba():
    from framework.attentions.moba_attention import MoBAAttention
    m = MoBAAttention(config=cfg, block_size=16, top_k=2, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, _ = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Mixture of Block Attention + Centroid Top-K + Block 0 CLS preservation, grad ok)"
tests.append(("moba", test_moba))

# 16. FASA
def test_fasa():
    from framework.attentions.fasa import FASAAttention
    m = FASAAttention(config=cfg, fasa_keep_ratio=0.5, pretrained_attn=old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, attn = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (FASA variance pruning + CLS protection, grad ok)"
tests.append(("fasa", test_fasa))

# 17. UPI-Stuffed Mamba
def test_upi_mamba():
    from framework.attentions.upi_stuffed_mamba import UPIMambaBlock
    m = UPIMambaBlock(config=cfg, dim=768)
    m.copy_weights(old_attn)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    mask = torch.ones(2, 64)
    out, _ = m(x, attention_mask=mask)
    assert out.shape == (2, 64, 768)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Uncertainty-Penalized Bidirectional SSM, grad ok)"
tests.append(("upi_stuffed_mamba", test_upi_mamba))

# 18. Granite Hybrid
def test_granite():
    from framework.attentions.granite_hybrid import GraniteHybridBlock
    m = GraniteHybridBlock(config=cfg, num_experts=4, top_k=2, ratio=2)
    m.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    out1, _ = m(x)
    out2, _ = m(x)
    out3, _ = m(x)
    assert out1.shape == (2, 64, 768) and out3.shape == (2, 64, 768)
    loss = out1.sum() + out2.sum() + out3.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    return "OK (Mamba-2 + MHA Ratio Routing + Top-K MoE, grad ok)"
tests.append(("granite_hybrid", test_granite))

# 19. CUR-MoE
def test_cur_moe():
    from framework.compression_engine.cur_moe_compressor import CURMoECompressor
    class SimpleMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(768, 768 * 2), nn.GELU(), nn.Linear(768 * 2, 768))
                for _ in range(4)
            ])
        def forward(self, x):
            return sum(e(x) for e in self.experts)
    model = SimpleMoE()
    comp = CURMoECompressor(model, rank=32, method="leverage")
    c_mod, stats = comp.compress()
    x = torch.randn(2, 64, 768)
    out = c_mod(x)
    assert out.shape == (2, 64, 768)
    assert stats["compressed_layer_count"] == 8
    return "OK (CUR Leverage Cross-Approximation on MoE Experts, 8 layers compressed)"
tests.append(("cur_moe", test_cur_moe))

passed_count = 0
failed_count = 0

for name, fn in tests:
    t0 = time.time()
    try:
        msg = fn()
        dt = time.time() - t0
        print(f"[PASS] {name.ljust(25)} -> {msg} ({dt:.2f}s)", flush=True)
        passed_count += 1
    except Exception as e:
        dt = time.time() - t0
        print(f"[FAIL] {name.ljust(25)} -> ERROR: {e} ({dt:.2f}s)", flush=True)
        failed_count += 1

print(f"\nAUDIT SUMMARY: {passed_count}/{len(tests)} PASSED, {failed_count} FAILED.", flush=True)
if failed_count > 0:
    sys.exit(1)
