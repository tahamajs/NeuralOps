import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.attention_utils import inject_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ds = load_from_disk("Codes/framework/data/glue_data/sst2")
val = ds["validation"]
tok = AutoTokenizer.from_pretrained("models/roberta-base")

m1 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
ckpt = torch.load("results/unified_asvd/roberta/finetuned_model.pt", map_location=device)
m1.load_state_dict(ckpt, strict=False)
m1.eval()

m2 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
m2.load_state_dict(ckpt, strict=False)
inject_attention(m2, "adaptive_span", rank=16)
m2.eval()

enc = tok(val[:2]["sentence"], padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(device)
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

emb1 = m1.roberta.embeddings(input_ids)
ext_mask = m1.roberta.get_extended_attention_mask(attention_mask, input_ids.shape)

l1 = m1.roberta.encoder.layer[0]
l2 = m2.roberta.encoder.layer[0]
s1 = l1.attention.self
s2 = l2.attention.self
w2 = s2._wrapped_attention

with torch.no_grad():
    out_s1 = s1(emb1, attention_mask=ext_mask)[0]
    out_w2 = w2(emb1, attention_mask=ext_mask)[0]

print("Seq 0 length (valid tokens):", attention_mask[0].sum().item())
print("Valid tokens [0, :13] diff:", (out_s1[0, :13] - out_w2[0, :13]).abs().max().item())
print("Padded tokens [0, 13:] diff:", (out_s1[0, 13:] - out_w2[0, 13:]).abs().max().item())
print("out_s1[0, 13, :5]:", out_s1[0, 13, :5])
print("out_w2[0, 13, :5]:", out_w2[0, 13, :5])

# Now check what RobertaAttention (self + output) does:
a1 = l1.attention
a2 = l2.attention
with torch.no_grad():
    out_a1 = a1(emb1, attention_mask=ext_mask)[0]
    out_a2 = a2(emb1, attention_mask=ext_mask)[0]

print("Attention 0 valid tokens diff:", (out_a1[0, :13] - out_a2[0, :13]).abs().max().item())
print("Attention 0 padded tokens diff:", (out_a1[0, 13:] - out_a2[0, 13:]).abs().max().item())

# Now check Layer 0 (attention + intermediate + output)
with torch.no_grad():
    out_l1 = l1(emb1, attention_mask=ext_mask)[0]
    out_l2 = l2(emb1, attention_mask=ext_mask)[0]

print("Layer 0 valid tokens diff:", (out_l1[0, :13] - out_l2[0, :13]).abs().max().item())
print("Layer 0 padded tokens diff:", (out_l1[0, 13:] - out_l2[0, 13:]).abs().max().item())

# Now check through all 12 layers:
x1 = emb1
x2 = emb1
for i in range(12):
    layer1 = m1.roberta.encoder.layer[i]
    layer2 = m2.roberta.encoder.layer[i]
    with torch.no_grad():
        x1 = layer1(x1, attention_mask=ext_mask)[0]
        x2 = layer2(x2, attention_mask=ext_mask)[0]
    print(f"Layer {i:2d} valid tokens max diff: {(x1[0, :13] - x2[0, :13]).abs().max().item():.6e} | padded max diff: {(x1[0, 13:] - x2[0, 13:]).abs().max().item():.6e}")

# Check pooled output & classifier:
with torch.no_grad():
    logits1 = m1.classifier(x1)
    logits2 = m2.classifier(x2)

print("logits1:", logits1[0])
print("logits2:", logits2[0])
print("logits diff:", (logits1 - logits2).abs().max().item())
