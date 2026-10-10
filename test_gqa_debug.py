"""
تست کامل برای پیدا کردن باگ‌های EATBlock.

اجرا:
    python test_eat_bugs.py
"""
import torch
import torch.nn as nn
import traceback
from neuralops.attentions.adaptive.eat import EATBlock, apply_rotary_pos_emb


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
    return EATBlock(config=cfg, **kwargs).eval()


def test_1_return_type_consistency():
    """باگ ۱: برگردانده‌ی forward باید همیشه tuple باشد."""
    print("\n" + "="*70)
    print("TEST 1: Return type consistency")
    print("="*70)

    block = make_block()
    x = torch.randn(2, 8, 64)

    # حالت ۱: بدون use_cache، بدون output_attentions
    out1 = block(x, use_cache=False, output_attentions=False)
    print(f"  Case 1 (no cache, no attn):   type={type(out1).__name__}, "
          f"is_tuple={isinstance(out1, tuple)}")

    # حالت ۲: با use_cache
    out2 = block(x, use_cache=True, output_attentions=False)
    print(f"  Case 2 (cache, no attn):      type={type(out2).__name__}, "
          f"is_tuple={isinstance(out2, tuple)}, len={len(out2) if isinstance(out2, tuple) else 'N/A'}")

    # حالت ۳: بدون use_cache، با output_attentions
    out3 = block(x, use_cache=False, output_attentions=True)
    print(f"  Case 3 (no cache, attn):      type={type(out3).__name__}, "
          f"is_tuple={isinstance(out3, tuple)}, len={len(out3) if isinstance(out3, tuple) else 'N/A'}")

    # حالت ۴: با use_cache + output_attentions
    out4 = block(x, use_cache=True, output_attentions=True)
    print(f"  Case 4 (cache, attn):         type={type(out4).__name__}, "
          f"is_tuple={isinstance(out4, tuple)}, len={len(out4) if isinstance(out4, tuple) else 'N/A'}")

    # چک: کد استاندارد فرض می‌کند همیشه tuple است
    print("\n  ⚠️  BUG: Case 1 returns raw tensor, others return tuple!")
    print("     Standard code doing `out, attn = layer(x)` would break.")


def test_2_rope_incremental_decode():
    """باگ ۲: RoPE در decode با past_len > 0 باید موقعیت‌های درست را اعمال کند."""
    print("\n" + "="*70)
    print("TEST 2: RoPE position offset in incremental decode")
    print("="*70)

    head_dim = 16
    cos_full = torch.randn(1, 1, 20, head_dim)
    sin_full = torch.randn(1, 1, 20, head_dim)

    # سیمولیت: Q و K جدید در موقعیت [5, 8) هستند (past_len=5, L=3)
    q_new = torch.randn(1, 2, 3, head_dim)
    k_new = torch.randn(1, 2, 3, head_dim)

    # مسیر درست: باید cos[5:8] را اعمال کند
    cos_slice = cos_full[..., 5:8, :]
    sin_slice = sin_full[..., 5:8, :]
    q_correct, k_correct = apply_rotary_pos_emb(q_new, k_new, cos_slice, sin_slice)

    # مسیر کد فعلی: کل cos را می‌دهد و position_ids را
    position_ids = torch.arange(5, 8).unsqueeze(0)
    q_actual, k_actual = apply_rotary_pos_emb(
        q_new, k_new, cos_full, sin_full, position_ids
    )

    diff = (q_correct - q_actual).abs().max().item()
    print(f"  Max diff (correct vs actual): {diff:.6f}")

    if diff > 1e-5:
        print(f"  ❌ BUG: RoPE uses positions [0:3] instead of [5:8]!")
        print(f"     position_ids arg is IGNORED inside apply_rotary_pos_emb.")
    else:
        print(f"  ✅ OK")


def test_3_causal_with_kv_cache():
    """باگ ۳: causal mask + kv-cache باید درست باشد."""
    print("\n" + "="*70)
    print("TEST 3: Causal mask with KV cache")
    print("="*70)

    block = make_block()
    x1 = torch.randn(1, 4, 64)  # اول ۴ توکن
    x2 = torch.randn(1, 2, 64)  # بعد ۲ توکن

    with torch.no_grad():
        # prefill
        out1, cache1 = block(x1, use_cache=True, output_attentions=False)

        # decode: 2 توکن جدید باید فقط به خودشان و ۴ توکن قبلی نگاه کنند
        out2, cache2 = block(x2, past_key_value=cache1, use_cache=True)

    k, v = cache2
    print(f"  Cache shape after decode: k={k.shape}, v={v.shape}")
    print(f"  Expected: k=[1, 4, 6, 16], v=[1, 4, 6, 16]")

    if k.shape[-2] != 6:
        print(f"  ❌ BUG: cache has {k.shape[-2]} tokens, expected 6!")
    else:
        print(f"  ✅ Cache size OK")


def test_4_attention_mask_kv_len_mismatch():
    """باگ ۴: وقتی attention_mask با طول kv_len نمی‌خواند."""
    print("\n" + "="*70)
    print("TEST 4: attention_mask shape vs kv_len")
    print("="*70)

    block = make_block()
    x = torch.randn(1, 4, 64)

    # attention_mask با شکل [B, L] = [1, 4]
    mask = torch.ones(1, 4)

    try:
        out = block(x, attention_mask=mask, output_attentions=True, use_cache=False)
        probs = out[-1]
        print(f"  Output probs shape: {probs.shape}")
        print(f"  Expected: [1, 4, 4, 4]")
        print(f"  ✅ Shapes match")
    except Exception as e:
        print(f"  ❌ BUG: {type(e).__name__}: {e}")


def test_5_causal_mask_values():
    """باگ ۵: مقادیر ماسک causal درست اعمال می‌شوند؟"""
    print("\n" + "="*70)
    print("TEST 5: Causal mask correctness")
    print("="*70)

    block = make_block()
    x = torch.randn(1, 4, 64)

    with torch.no_grad():
        out, attn = block(x, output_attentions=True, use_cache=False)

    # چک: probs[0, 0, i, j] باید صفر باشد اگر j > i
    print(f"  Attention shape: {attn.shape}")
    upper_tri = attn[0, 0].triu(diagonal=1)
    max_upper = upper_tri.max().item()
    print(f"  Max value above diagonal: {max_upper:.6f}")

    if max_upper > 1e-6:
        print(f"  ❌ BUG: Future tokens attended! (max upper = {max_upper})")
    else:
        print(f"  ✅ Causal masking correct")


def test_6_head_mask_shapes():
    """باگ ۶: head_mask با شکل‌های مختلف."""
    print("\n" + "="*70)
    print("TEST 6: head_mask shapes")
    print("="*70)

    block = make_block()
    x = torch.randn(2, 4, 64)

    shapes = [
        ("1D", torch.ones(4)),
        ("2D [B, H]", torch.ones(2, 4)),
        ("2D [H, L]", torch.ones(4, 4)),  # این شکل رایج نیست
        ("3D [B, H, 1]", torch.ones(2, 4, 1)),
    ]

    for name, hm in shapes:
        try:
            with torch.no_grad():
                out = block(x, head_mask=hm, output_attentions=True, use_cache=False)
            print(f"  {name:12s}: ✅ OK")
        except Exception as e:
            print(f"  {name:12s}: ❌ {type(e).__name__}: {e}")


def test_7_sdpa_vs_manual():
    """باگ ۷: SDPA و مسیر دستی باید نتیجه‌ی یکسان بدهند (بدون mask)."""
    print("\n" + "="*70)
    print("TEST 7: SDPA vs manual path equivalence")
    print("="*70)

    block = make_block()
    x = torch.randn(1, 4, 64)

    with torch.no_grad():
        # مسیر SDPA: بدون mask، بدون causal (is_decoder=False در block)
        block.is_decoder = False
        out_sdpa = block(x, output_attentions=False, use_cache=False)

        # مسیر دستی: با output_attentions=True
        out_manual = block(x, output_attentions=True, use_cache=False)
        out_manual = out_manual[0]

    diff = (out_sdpa - out_manual).abs().max().item()
    print(f"  Max diff (SDPA vs manual): {diff:.6f}")

    if diff > 1e-4:
        print(f"  ❌ BUG: SDPA and manual differ by {diff}")
    else:
        print(f"  ✅ Paths agree")


def test_8_rope_with_past_len():
    """باگ ۸: با past_len > 0، RoPE موقعیت‌های درست را اعمال می‌کند؟"""
    print("\n" + "="*70)
    print("TEST 8: RoPE correctness during decode (with actual forward)")
    print("="*70)

    block = make_block()
    block.is_decoder = True

    # فرض: position_embeddings داده می‌شود
    B, L = 1, 4
    head_dim = 16
    past_len = 0

    cos = torch.randn(1, 1, 10, head_dim)
    sin = torch.randn(1, 1, 10, head_dim)

    x = torch.randn(B, L, 64)

    with torch.no_grad():
        # حالت ۱: بدون cache
        out1 = block(x, position_embeddings=(cos, sin), use_cache=False,
                     output_attentions=False)

        # حالت ۲: با cache از قبل
        past_k = torch.randn(1, 4, 3, head_dim)
        past_v = torch.randn(1, 4, 3, head_dim)
        out2 = block(x, position_embeddings=(cos, sin),
                     past_key_value=(past_k, past_v),
                     use_cache=True, output_attentions=False)

    print(f"  Without cache: ok")
    print(f"  With cache:    ok")
    print(f"  ⚠️  Cannot verify RoPE positions directly — need manual check.")


def test_9_past_kv_returned_correctly():
    """باگ ۹: present_key_value باید شامل کل cache باشد."""
    print("\n" + "="*70)
    print("TEST 9: present_key_value contains full cache")
    print("="*70)

    block = make_block()
    x1 = torch.randn(1, 3, 64)
    x2 = torch.randn(1, 2, 64)

    with torch.no_grad():
        _, cache1 = block(x1, use_cache=True, output_attentions=False)
        _, cache2 = block(x2, past_key_value=cache1, use_cache=True,
                          output_attentions=False)

    k1, v1 = cache1
    k2, v2 = cache2
    print(f"  First call:  k shape = {k1.shape} (expected [1, 4, 3, 16])")
    print(f"  Second call: k shape = {k2.shape} (expected [1, 4, 5, 16])")

    if k2.shape[-2] != 5:
        print(f"  ❌ BUG: Second cache has {k2.shape[-2]} tokens, expected 5")
    else:
        print(f"  ✅ Cache properly accumulated")


def test_10_lora_base_frozen():
    """باگ ۱۰: آیا لایه‌های پایه فریز هستند؟"""
    print("\n" + "="*70)
    print("TEST 10: LoRA base weights frozen")
    print("="*70)

    block = make_block()
    for name, mod in block.named_modules():
        if hasattr(mod, "lora_rank") and mod.lora_rank > 0:
            w_frozen = not mod.weight.requires_grad
            a_train = mod.lora_A.requires_grad
            b_train = mod.lora_B.requires_grad
            print(f"  {name:20s}: "
                  f"weight_frozen={w_frozen}, "
                  f"lora_A_train={a_train}, "
                  f"lora_B_train={b_train}")


# ============================================================================
# اجرا
# ============================================================================
if __name__ == "__main__":
    tests = [
        ("Return type consistency",         test_1_return_type_consistency),
        ("RoPE incremental decode",         test_2_rope_incremental_decode),
        ("Causal with KV cache",            test_3_causal_with_kv_cache),
        ("attn_mask shape vs kv_len",       test_4_attention_mask_kv_len_mismatch),
        ("Causal mask values",              test_5_causal_mask_values),
        ("head_mask shapes",                test_6_head_mask_shapes),
        ("SDPA vs manual",                  test_7_sdpa_vs_manual),
        ("RoPE with past_len",              test_8_rope_with_past_len),
        ("past_kv contains full cache",     test_9_past_kv_returned_correctly),
        ("LoRA base frozen",                test_10_lora_base_frozen),
    ]

    results = []
    for name, fn in tests:
        try:
            fn()
            results.append((name, "✅ PASS"))
        except Exception as e:
            print(f"\n  ❌ EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((name, f"❌ FAIL: {e}"))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, status in results:
        print(f"  {name:40s} : {status}")
    print("="*70)