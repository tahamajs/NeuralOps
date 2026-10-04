import torch
from transformers import GPT2LMHeadModel
from neuralops.core.registry import ATTENTION_REGISTRY

print("=" * 60)
print("jamba_info in registry:", "jamba_info" in ATTENTION_REGISTRY)
print("=" * 60)

from neuralops.eval_all_models_datasets import apply_module_to_model
print("[OK] apply_module_to_model imported")

model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Before injection:")
print("  transformer.h.0.attn class:", model.transformer.h[0].attn.__class__.__name__)

model = apply_module_to_model(model, "jamba_info", "attention", rank=64, device="cpu")[0]

print("After injection:")
print("  transformer.h.0.attn class:", model.transformer.h[0].attn.__class__.__name__)

from neuralops.benchmarks.kd_quick_test import set_train_scope
n = set_train_scope(model, "module")
print(f"Train scope='module': {n} trainable params")

trainable_names = [nm for nm, p in model.named_parameters() if p.requires_grad]
print(f"Total trainable tensors: {len(trainable_names)}")
for name in trainable_names[:20]:
    print("   ", name)

print("\nFirst 10 non-trainable names:")
nontrainable = [nm for nm, p in model.named_parameters() if not p.requires_grad]
for name in nontrainable[:10]:
    print("   ", name)
