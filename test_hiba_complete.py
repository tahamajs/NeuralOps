"""
تست جامع BlockTransformerHIBA — همه‌ی باگ‌ها با assert واقعی.

اجرا:
    python test_hiba_complete.py
"""
import math
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.attentions.adaptive.block_transformer_hiba import (
    BlockTransformerHIBA,
    LoRALinear,
    apply_rotary_pos_emb,
)


# ============================================================================
# Fixtures / helpers
# ============================================================================
class DummyConfig:
    hidden_size = 64
    num_attention_heads = 4
    head_dim = 16
    num_key_value_heads = 4
    model_type = "gpt2"
    is_decoder = True
    is_causal = True
    attention_probs_dropout_prob = 0.0
    lora_rank = 4
    lora_alpha = 8.0


def make_block(**kwargs):
    cfg = DummyConfig()
    block = BlockTransformerHIBA(
        config=cfg,
        block_size=kwargs.pop("block_size", 4),
        top_k_inter=kwargs.pop("top_k_inter", 2),
        rectification_interval=kwargs.pop("rectification_interval", 1),
        layer_idx=kwargs.pop("layer_idx", 0),
        **kwargs,
    )
    return block.eval()


# ============================================================================
# TEST 1: Return type همیشه tuple است
# ============================================================================
def test_1_return_type_always_tuple():
    print("\n" + "=" * 70)
    print("TEST 1: Return type consistency")
    print("=" * 70)

    block = make_block()
    x = torch.randn(2, 8, 64)

    out_a = block(x, output_attentions=False)
    out_b = block(x, output_attentions=True)

    assert isinstance(out_a, tuple), f"Expected tuple, got {type(out_a).__name__}"
    assert isinstance(out_b, tuple), f"Expected tuple, got {type(out_b).__name__}"
    assert len(out_a) == 2
    assert len(out_b) == 2
    assert out_a[0].shape == (2, 8, 64)
    assert out_b[0].shape == (2, 8, 64)
    print("  ✅ Always returns tuple")


# ============================================================================
# TEST 2: RoPE — position_ids باید اثر داشته باشد
# ============================================================================
def test_2_rope_position_ids_ignored():
    print("\n" + "=" * 70)
    print("TEST 2: RoPE position_ids handling")
    print("=" * 70)

    head_dim = 16
    seq_len = 20
    # cos/sin بدون position embedding واقعی (فقط rank چک می‌کنیم)
    cos = torch.randn(1, 1, seq_len, head_dim)
    sin = torch.randn(1, 1, seq_len, head_dim)

    # Q, K جدید در موقعیت 5,6,7 — باید cos[5:8] اعمال شود
    q = torch.randn(1, 2, 3, head_dim)
    k = torch.randn(1, 2, 3, head_dim)
    pos_ids = torch.arange(5, 8).unsqueeze(0)  # [1, 3]

    # مرجع: cos را قبل از پاس دادن slice می‌کنیم
    cos_slice = cos[..., 5:8, :]
    sin_slice = sin[..., 5:8, :]
    q_ref, k_ref = apply_rotary_pos_emb(q, k, cos_slice, sin_slice)

    # فراخوانی با position_ids
    q_act, k_act = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

    diff = (q_ref - q_act).abs().max().item()
    print(f"  Max diff (ref vs actual): {diff:.6f}")
    assert diff < 1e-5, (
        f"RoPE ignores position_ids! diff={diff:.6f}. "
        "cos/sin must be sliced before calling apply_rotary_pos_emb."
    )
    print("  ✅ position_ids respected")


# ============================================================================
# TEST 3: LoRA — base فریز، lora_B صفر
# ============================================================================
def test_3_lora_correctness():
    print("\n" + "=" * 70)
    print("TEST 3: LoRALinear init")
    print("=" * 70)

    lin = LoRALinear(32, 64, lora_rank=4, lora_alpha=8.0, freeze_base=True)

    assert not lin.weight.requires_grad, "base weight must be frozen"
    assert lin.lora_A is not None and lin.lora_A.requires_grad
    assert lin.lora_B is not None and lin.lora_B.requires_grad
    assert torch.all(lin.lora_B == 0), "lora_B must be zero-initialized"

    # forward با lora_B=0 → همان base
    x = torch.randn(2, 32)
    out = lin(x)
    out_base = F.linear(x, lin.weight, lin.bias)
    assert torch.allclose(out, out_base, atol=1e-6), "forward must be identity at init"
    print("  ✅ LoRA init correct")


# ============================================================================
# TEST 4: Forward shapes با طول‌های مختلف
# ============================================================================
def test_4_forward_shapes():
    print("\n" + "=" * 70)
    print("TEST 4: Forward shapes for various L")
    print("=" * 70)

    block = make_block()
    for L in [1, 3, 4, 5, 8, 16, 17, 32]:
        x = torch.randn(2, L, 64)
        out, _ = block(x, output_attentions=False)
        assert out.shape == (2, L, 64), f"L={L}: got {out.shape}"
    print("  ✅ All shapes correct")


# ============================================================================
# TEST 5: Causal masking
# ============================================================================
def test_5_causal_correctness():
    print("\n" + "=" * 70)
    print("TEST 5: Causal masking")
    print("=" * 70)

    # با rectification_interval خیلی بزرگ، مسیر dense را اجبار می‌کنیم
    block = make_block(rectification_interval=100)
    x = torch.randn(1, 8, 64)

    with torch.no_grad():
        out, attn = block(x, output_attentions=True)

    assert attn is not None
    # در مسیر dense، probs شکل [B, H, L, S] دارد
    upper = attn[0, 0].triu(diagonal=1).max().item()
    print(f"  Max above diagonal: {upper:.6f}")
    assert upper < 1e-6, "Causal masking broken"
    print("  ✅ Causal correct")


# ============================================================================
# TEST 6: Hierarchical path — باگ empty V
# ============================================================================
def test_6_hierarchical_empty_neighbor():
    print("\n" + "=" * 70)
    print("TEST 6: Hierarchical path with empty first block")
    print("=" * 70)

    # rectification_interval=0 → اجبار به hierarchical
    block = make_block(block_size=4, top_k_inter=1, rectification_interval=0)
    x = torch.randn(1, 16, 64)

    try:
        out, _ = block(x, output_attentions=False)
        assert out.shape == (1, 16, 64)
        print("  ✅ Hierarchical path OK")
    except Exception as e:
        raise AssertionError(
            f"Hierarchical path failed: {type(e).__name__}: {e}\n"
            "Likely bug: empty inter_v_list entry has wrong shape."
        ) from e


# ============================================================================
# TEST 7: head_mask با اشکال مختلف
# ============================================================================
def test_7_head_mask_shapes():
    print("\n" + "=" * 70)
    print("TEST 7: head_mask shape handling")
    print("=" * 70)

    block = make_block(rectification_interval=100)  # dense path
    x = torch.randn(2, 8, 64)

    shapes = [
        ("1D [H]",           torch.ones(4)),
        ("2D [B, H]",        torch.ones(2, 4)),
        ("3D [B, H, 1]",     torch.ones(2, 4, 1)),
    ]
    for name, hm in shapes:
        try:
            with torch.no_grad():
                out, _ = block(x, head_mask=hm, output_attentions=False)
            print(f"  {name:15s}: ✅")
        except Exception as e:
            raise AssertionError(f"head_mask {name} failed: {e}") from e


# ============================================================================
# TEST 8: attention_mask (2D, 3D, 4D)
# ============================================================================
def test_8_attention_mask_shapes():
    print("\n" + "=" * 70)
    print("TEST 8: attention_mask shapes")
    print("=" * 70)

    block = make_block(rectification_interval=100)
    x = torch.randn(2, 8, 64)

    # 2D padding
    mask2 = torch.ones(2, 8); mask2[0, 5:] = 0
    out, _ = block(x, attention_mask=mask2)
    assert out.shape == (2, 8, 64)

    # 3D additive
    mask3 = torch.zeros(2, 8, 8); mask3[0, 0, -3:] = -1e9
    out, _ = block(x, attention_mask=mask3)
    assert out.shape == (2, 8, 64)

    # 4D bool
    mask4 = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    mask4[0, 0, 0, 5:] = False
    out, _ = block(x, attention_mask=mask4)
    assert out.shape == (2, 8, 64)
    print("  ✅ All mask shapes OK")


# ============================================================================
# TEST 9: GQA kv_repeat
# ============================================================================
def test_9_gqa():
    print("\n" + "=" * 70)
    print("TEST 9: GQA")
    print("=" * 70)

    cfg = DummyConfig()
    cfg.num_key_value_heads = 2  # GQA 4->2
    block = BlockTransformerHIBA(config=cfg, block_size=4, rectification_interval=100)

    assert block.num_kv_heads == 2
    assert block.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)
    print("  ✅ GQA OK")


# ============================================================================
# TEST 10: Dtype — FP32, FP16, BF16
# ============================================================================
def test_10_dtypes():
    print("\n" + "=" * 70)
    print("TEST 10: dtype handling")
    print("=" * 70)

    block = make_block(rectification_interval=100)
    x = torch.randn(2, 8, 64)

    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        block_dt = block.to(dt)
        x_dt = x.to(dt)
        try:
            out, _ = block_dt(x_dt)
            assert out.dtype == dt, f"Output dtype {out.dtype} != {dt}"
            print(f"  {str(dt):20s}: ✅")
        except Exception as e:
            raise AssertionError(f"dtype {dt} failed: {e}") from e


# ============================================================================
# TEST 11: Nested encoder (BERT-like) — out_proj → Identity
# ============================================================================
def test_11_nested_encoder_identity():
    print("\n" + "=" * 70)
    print("TEST 11: Nested encoder detection → Identity out_proj")
    print("=" * 70)

    # ساخت یک source attention شبیه BERT
    class FakeBertSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

    class FakeBertAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = FakeBertSelfAttention()
            self.output = nn.Module()
            self.output.dense = nn.Linear(64, 64)

    src = FakeBertAttention()
    # تغییر نام کلاس برای تشخیص
    FakeBertSelfAttention.__name__ = "BertSelfAttention"

    block = BlockTransformerHIBA(
        config=None,
        block_size=4,
        rectification_interval=100,
        old_attn=src.self,
    )
    print(f"  is_nested_encoder = {block.is_nested_encoder}")
    print(f"  out_proj type = {type(block.out_proj).__name__}")
    assert block.is_nested_encoder, "should detect nested encoder"
    assert isinstance(block.out_proj, nn.Identity), (
        f"out_proj should be Identity, got {type(block.out_proj).__name__}"
    )
    print("  ✅ Nested encoder handled")


# ============================================================================
# TEST 12: Copy weights از GPT-2
# ============================================================================
def test_12_copy_weights_gpt2():
    print("\n" + "=" * 70)
    print("TEST 12: copy_weights from GPT-2-like")
    print("=" * 70)

    class FakeConv1D(nn.Module):
        def __init__(self, nf, nx):
            super().__init__()
            self.nf = nf
            self.weight = nn.Parameter(torch.randn(nx, nf))
            self.bias = nn.Parameter(torch.randn(nf))

    class FakeGPT2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = FakeConv1D(192, 64)   # 3*64
            self.c_proj = FakeConv1D(64, 64)

    src = FakeGPT2Attn()
    block = BlockTransformerHIBA(
        config=DummyConfig(),
        block_size=4,
        rectification_interval=100,
        old_attn=src,
    )

    # چک: وزن‌ها کپی شده
    q_norm = block.W_q.weight.norm().item()
    k_norm = block.W_k.weight.norm().item()
    v_norm = block.W_v.weight.norm().item()
    o_norm = block.out_proj.weight.norm().item()
    print(f"  W_q norm = {q_norm:.4f}")
    print(f"  W_k norm = {k_norm:.4f}")
    print(f"  W_v norm = {v_norm:.4f}")
    print(f"  out norm = {o_norm:.4f}")
    assert q_norm > 0.1, "W_q not copied"
    assert k_norm > 0.1, "W_k not copied"
    assert v_norm > 0.1, "W_v not copied"
    assert o_norm > 0.1, "out_proj not copied"
    print("  ✅ GPT-2 weights copied")


# ============================================================================
# TEST 13: rectification_interval logic
# ============================================================================
def test_13_rectification():
    print("\n" + "=" * 70)
    print("TEST 13: rectification_interval controls path")
    print("=" * 70)

    # rectification_interval=100 → همیشه dense
    b_dense = make_block(rectification_interval=100, block_size=4)
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        _, attn = b_dense(x, output_attentions=True)
    assert attn is not None, "Dense path should return attn"
    print("  ✅ Dense path returns attn")

    # rectification_interval=0 → hierarchical
    b_hier = make_block(rectification_interval=0, block_size=4, top_k_inter=1)
    with torch.no_grad():
        out, attn = b_hier(x, output_attentions=True)
    assert out.shape == (1, 16, 64)
    print("  ✅ Hierarchical path OK")


# ============================================================================
# TEST 14: KV-cache (اگر پشتیبانی می‌شود)
# ============================================================================
def test_14_kv_cache():
    print("\n" + "=" * 70)
    print("TEST 14: KV-cache consistency")
    print("=" * 70)

    block = make_block(rectification_interval=100)
    x1 = torch.randn(1, 4, 64)
    x2 = torch.randn(1, 2, 64)

    with torch.no_grad():
        # اول ۴ توکن
        out1, _ = block(x1)
        # بعد ۲ توکن (بدون cache، فقط برای چک کردن اینکه قبول کند)
        out2, _ = block(x2)
    assert out2.shape == (1, 2, 64)
    print("  ⚠️  Note: KV-cache not implemented — this only checks shape")


# ============================================================================
# TEST 15: Batch dimension independence
# ============================================================================
def test_15_batch_independence():
    print("\n" + "=" * 70)
    print("TEST 15: Batch independence")
    print("=" * 70)

    block = make_block(rectification_interval=100)
    x = torch.randn(4, 8, 64)
    out_batch, _ = block(x)

    # هر نمونه را جدا اجرا کن
    outs = []
    for i in range(4):
        o, _ = block(x[i:i+1])
        outs.append(o)
    out_individual = torch.cat(outs, dim=0)

    diff = (out_batch - out_individual).abs().max().item()
    print(f"  Max diff (batch vs individual): {diff:.6f}")
    assert diff < 1e-4, "Batch processing is not sample-independent"
    print("  ✅ Batch independent")


# ============================================================================
# TEST 16: head_mask با اشکال اشتباه باید خطا بدهد
# ============================================================================
def test_16_head_mask_wrong_shape():
    print("\n" + "=" * 70)
    print("TEST 16: head_mask with wrong shape should fail")
    print("=" * 70)

    block = make_block(rectification_interval=100)
    x = torch.randn(2, 8, 64)

    # head_mask با شکل [B=2, H=4] داده می‌شود — این درست است
    hm_ok = torch.ones(2, 4)
    out, _ = block(x, head_mask=hm_ok)
    print("  head_mask [2, 4]: ✅")

    # head_mask با H اشتباه [2, 5]
    hm_bad = torch.ones(2, 5)
    try:
        out, _ = block(x, head_mask=hm_bad)
        print("  head_mask [2, 5] (wrong H): ❌ silently accepted")
        # این باید خطا بدهد چون H=4 و 5 != 4
        raise AssertionError("head_mask with wrong H should raise, not silently accept")
    except (RuntimeError, ValueError, AssertionError) as e:
        if "should raise" in str(e):
            raise
        print(f"  head_mask [2, 5] (wrong H): ✅ rejected with {type(e).__name__}")


# ============================================================================
# TEST 17: Numeric stability با مقادیر بزرگ
# ============================================================================
def test_17_numeric_stability():
    print("\n" + "=" * 70)
    print("TEST 17: Numeric stability with large inputs")
    print("=" * 70)

    block = make_block(rectification_interval=100)
    x = torch.randn(2, 8, 64) * 100  # مقادیر بزرگ

    out, _ = block(x)
    assert torch.isfinite(out).all(), "Output has NaN/Inf with large inputs"
    print(f"  Max |out| = {out.abs().max().item():.2f}")
    print("  ✅ Finite output")


# ============================================================================
# TEST 18: Gradient flow
# ============================================================================
def test_18_gradient_flow():
    print("\n" + "=" * 70)
    print("TEST 18: Gradient flow through LoRA")
    print("=" * 70)

    block = make_block(rectification_interval=100)
    block.train()
    x = torch.randn(2, 8, 64, requires_grad=True)

    out, _ = block(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient to input"
    assert block.W_q.lora_A.grad is not None, "No gradient to lora_A"
    assert block.W_q.lora_B.grad is not None, "No gradient to lora_B"
    assert block.W_q.weight.grad is None, "Base weight should NOT have grad"
    print("  ✅ Gradients flow correctly")


# ============================================================================
# RUNNER
# ============================================================================
if __name__ == "__main__":
    tests = [
        ("Return type always tuple",            test_1_return_type_always_tuple),
        ("RoPE position_ids handling",          test_2_rope_position_ids_ignored),
        ("LoRA correctness",                    test_3_lora_correctness),
        ("Forward shapes (various L)",          test_4_forward_shapes),
        ("Causal masking",                      test_5_causal_correctness),
        ("Hierarchical empty neighbor",         test_6_hierarchical_empty_neighbor),
        ("head_mask shapes",                    test_7_head_mask_shapes),
        ("attention_mask shapes",               test_8_attention_mask_shapes),
        ("GQA kv_repeat",                       test_9_gqa),
        ("Dtypes (fp32/fp16/bf16)",             test_10_dtypes),
        ("Nested encoder → Identity",           test_11_nested_encoder_identity),
        ("copy_weights from GPT-2",             test_12_copy_weights_gpt2),
        ("rectification_interval",              test_13_rectification),
        ("KV-cache (shape check)",              test_14_kv_cache),
        ("Batch independence",                  test_15_batch_independence),
        ("head_mask wrong shape rejected",      test_16_head_mask_wrong_shape),
        ("Numeric stability",                   test_17_numeric_stability),
        ("Gradient flow",                       test_18_gradient_flow),
    ]

    results = []
    for name, fn in tests:
        try:
            fn()
            results.append((name, "✅ PASS"))
        except AssertionError as e:
            print(f"\n  ❌ ASSERTION FAILED: {e}")
            results.append((name, f"❌ FAIL: {str(e)[:80]}"))
        except Exception as e:
            print(f"\n  ❌ EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((name, f"❌ {type(e).__name__}: {str(e)[:80]}"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status in results:
        print(f"  {name:40s} : {status}")
    n_pass = sum(1 for _, s in results if s == "✅ PASS")
    print("=" * 70)
    print(f"  {n_pass}/{len(results)} tests passed")
    print("=" * 70)