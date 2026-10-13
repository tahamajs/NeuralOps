"""Task 4 — Integration test on 3 models."""
import sys
import traceback
import neuralops
from transformers import AutoModel

MODEL_NAMES = [
    "distilbert-base-uncased",
    "roberta-base",
    "gpt2-medium",
]

results = {}
for model_name in MODEL_NAMES:
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print("="*60)
    try:
        model = AutoModel.from_pretrained(model_name)
        print(f"  Loaded: {model.__class__.__name__}")
        model = neuralops.replace_attention(model, "svd_linear_attention", rank=64)
        print(f"  After replace_attention: OK")
        model = neuralops.compress(model, method="direct_svd", rank=32)
        print(f"  After compress: OK")
        results[model_name] = f"PASS ({model.__class__.__name__})"
    except Exception as exc:
        tb = traceback.format_exc()
        results[model_name] = f"FAIL: {exc}"
        print(f"  ERROR: {exc}")
        print(tb)

print("\n\n===== SUMMARY =====")
all_pass = True
for k, v in results.items():
    status = "✅" if v.startswith("PASS") else "❌"
    print(f"{status} {k}: {v}")
    if not v.startswith("PASS"):
        all_pass = False

sys.exit(0 if all_pass else 1)
