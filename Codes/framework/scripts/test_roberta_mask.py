import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("models/roberta-base")
m = AutoModelForSequenceClassification.from_pretrained("models/roberta-base").to(device)

enc = tok(["hello world", "this is a longer sentence for testing"], padding="max_length", max_length=128, return_tensors="pt").to(device)

captured = {}
orig_forward = m.roberta.encoder.layer[0].attention.self.forward

def hook(hidden_states, *args, **kwargs):
    captured["args"] = args
    captured["kwargs"] = kwargs
    return orig_forward(hidden_states, *args, **kwargs)

m.roberta.encoder.layer[0].attention.self.forward = hook

m(**enc)

print("kwargs keys:", list(captured["kwargs"].keys()))
msk = captured["kwargs"].get("attention_mask")
print("msk type:", type(msk))
if isinstance(msk, torch.Tensor):
    print("msk shape:", msk.shape, "dtype:", msk.dtype, "min:", msk.min().item(), "max:", msk.max().item())
    print("msk unique values:", torch.unique(msk[:1, 0, 0, :20]).tolist())
