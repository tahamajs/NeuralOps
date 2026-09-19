import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
import importlib
import inspect
import sys

MODEL_PATH = "models/distilbert_mnli_full"
config = AutoConfig.from_pretrained(MODEL_PATH)
print(f"Config: hidden_size={getattr(config, 'hidden_size', None)}, "
      f"dim={getattr(config, 'dim', None)}, "
      f"num_attention_heads={getattr(config, 'num_attention_heads', None)}, "
      f"n_layers={getattr(config, 'n_layers', None)}")
h_size = getattr(config, "hidden_size", getattr(config, "dim", 768))
n_heads = getattr(config, "num_attention_heads", 12)
print(f"Calculated head_dim = {h_size // n_heads}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
layer0 = model.distilbert.transformer.layer[0]
attn = layer0.attention

# Report all projection shapes
print("\n--- DistilBERT Attention Projections ---")
for name in ["q_lin", "k_lin", "v_lin", "out_lin"]:
    m = getattr(attn, name, None)
    if m is not None and hasattr(m, "weight"):
        print(f"{name}: weight.shape = {m.weight.shape}, bias = {m.bias.shape if m.bias is not None else None}")

# Try to inspect each target failed module
FAIL_MODULES = [
    "cosformer", "hymba_transx", "infini_attention", "lsh_attention",
    "minimal_aurora", "minimal_svd", "nsa_attention", "nystrom_attention",
    "replace", "spectral_mamba",
]

print("\n--- Diagnosing Failed Attention Modules ---")
for mod_name in FAIL_MODULES:
    try:
        mod = importlib.import_module(f"framework.attentions.{mod_name}")
        inject_fn = None
        for attr in dir(mod):
            if attr.startswith("inject_"):
                inject_fn = getattr(mod, attr)
                break
        if inject_fn is None:
            print(f"{mod_name}: no inject_* found")
            continue
        print(f"\n=== {mod_name} ===")
        print(f"  inject_fn: {inject_fn.__name__}")

        # Clone fresh model for testing
        test_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        test_layer = test_model.distilbert.transformer.layer[0]
        try:
            res = inject_fn(layer=test_layer, model=test_model, model_config=config)
            print(f"  injection test result: {type(res)}")
        except Exception as inj_exc:
            print(f"  injection test EXCEPTION: {type(inj_exc).__name__}: {inj_exc}")
            continue

        # Test forward pass with batch of shape [2, 32]
        dummy_input = torch.randint(0, 1000, (2, 32))
        try:
            out = test_model(dummy_input)
            print(f"  forward pass SUCCESS: logits shape = {out.logits.shape}")
        except Exception as fwd_exc:
            print(f"  forward pass FAILURE: {type(fwd_exc).__name__}: {fwd_exc}")
    except Exception as e:
        print(f"{mod_name}: import error: {e}")
