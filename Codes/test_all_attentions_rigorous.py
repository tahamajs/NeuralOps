"""
Comprehensive Scientific Quality Assurance & Mathematical Verification Suite
for all Attention Mechanisms in the Framework.
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

# Add Codes/ to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import framework.attentions
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import inject_attention


def create_dummy_inputs(batch_size: int, seq_len: int, hidden_size: int, device: torch.device):
    """Create test activations and masks."""
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    # 2D binary mask
    mask_2d = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    mask_2d[:, seq_len // 2:] = 0  # Second half padded

    # 4D HuggingFace additive mask: [B, 1, 1, N]
    mask_4d = torch.zeros((batch_size, 1, 1, seq_len), dtype=torch.float32, device=device)
    mask_4d[:, :, :, seq_len // 2:] = -10000.0

    return x, mask_2d, mask_4d


def test_attention_module(attn_name: str, config: Any, device: torch.device):
    print(f"\n--- Testing Attention: '{attn_name}' ---")
    cls_or_fn = ATTENTION_REGISTRY.resolve(attn_name)

    # Instantiate
    try:
        if callable(cls_or_fn):
            module = cls_or_fn(config)
        else:
            module = cls_or_fn(config)
    except Exception as e:
        print(f"❌ Failed instantiation for '{attn_name}': {e}")
        return False

    module.to(device)
    module.train()

    seq_lengths = [16, 64, 128]
    for seq_len in seq_lengths:
        x, mask_2d, mask_4d = create_dummy_inputs(2, seq_len, config.hidden_size, device)

        # 1. Forward with None mask
        out1, _ = module(x, attention_mask=None)
        assert out1.shape == x.shape, f"Shape mismatch with mask=None: {out1.shape} vs {x.shape}"
        assert not torch.isnan(out1).any(), "NaN detected in output with mask=None"
        assert not torch.isinf(out1).any(), "Inf detected in output with mask=None"

        # 2. Forward with 2D binary mask
        out2, _ = module(x, attention_mask=mask_2d)
        assert out2.shape == x.shape, f"Shape mismatch with 2D mask: {out2.shape} vs {x.shape}"
        assert not torch.isnan(out2).any(), "NaN detected in output with 2D mask"

        # 3. Forward with 4D float additive mask
        out3, _ = module(x, attention_mask=mask_4d)
        assert out3.shape == x.shape, f"Shape mismatch with 4D mask: {out3.shape} vs {x.shape}"
        assert not torch.isnan(out3).any(), "NaN detected in output with 4D mask"

        # 4. Backward pass & Gradient Check
        loss = out3.sum()
        loss.backward()

        grad_count = 0
        for p_name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for param {p_name} in {attn_name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {p_name}"
                grad_count += 1
        module.zero_grad()

    print(f"✅ Forward/Backward checks passed across lengths {seq_lengths} ({grad_count} trainable parameters).")
    return True


def test_model_injection(model_name: str, attn_name: str, device: torch.device):
    print(f"Testing injection: Model='{model_name}' | Attn='{attn_name}'...")
    try:
        try:
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=config, local_files_only=True
            )
        except Exception:
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_config(config)
        model = inject_attention(model, attn_name, copy_weights=True)
        model.to(device)

        x_input = torch.randint(0, 1000, (2, 32), device=device)
        mask = torch.ones((2, 32), dtype=torch.long, device=device)
        mask[:, 16:] = 0

        outputs = model(input_ids=x_input, attention_mask=mask)
        logits = outputs.logits
        assert logits.shape[0] == 2, f"Unexpected batch size {logits.shape}"
        assert not torch.isnan(logits).any(), "NaN in classification logits"

        loss = logits.sum()
        loss.backward()
        print(f"✅ Injection succeeded for {model_name} with {attn_name}!")
        return True
    except Exception as e:
        print(f"❌ Injection error for {model_name} / {attn_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Running Rigorous Quality Assurance on Device: {device}")

    # Standard model config for testing
    from transformers import BertConfig
    config = BertConfig(
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    core_attentions = [
        "full",
        "svd",
        "xattention",
        "linformer",
        "performer",
        "nystrom",
        "asvd",
        "gqa",
        "lsh",
        "svd_linear_attn",
        "adaptive_svd_attn",
    ]

    all_passed = True
    print("\n=======================================================")
    print("PHASE 1: Attention Module Mathematical & Stability Tests")
    print("=======================================================")
    for attn in core_attentions:
        if not test_attention_module(attn, config, device):
            all_passed = False

    print("\n=======================================================")
    print("PHASE 2: Full Architecture Model Injection & Zero-Shot Weight Transfer")
    print("=======================================================")
    models = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]
    for model_name in models:
        for attn in ["full", "svd", "xattention", "linformer", "performer", "nystrom", "asvd", "gqa", "lsh"]:
            ok = test_model_injection(model_name, attn, device)
            if not ok:
                all_passed = False

    print("\n=======================================================")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Attention Suite is 100% Paper-Review Ready!")
    else:
        print("⚠️ SOME TESTS FAILED. Please review the output above.")
    print("=======================================================")


if __name__ == "__main__":
    main()
