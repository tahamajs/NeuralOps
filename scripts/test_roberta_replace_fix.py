import sys
sys.path.insert(0, 'Codes')
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.attentions.replace import ReplaceAttention

model = AutoModelForSequenceClassification.from_pretrained('models/roberta_sst2')
tok = AutoTokenizer.from_pretrained('models/roberta_sst2')
inputs = tok(['This movie was fantastic and brilliant!', 'Terrible, waste of time.'], return_tensors='pt', padding=True)

with torch.no_grad():
    orig_logits = model(**inputs).logits

print('Original logits:', orig_logits.tolist())

# Now inject ReplaceAttention correctly with cand (layer.attention)
for layer in model.roberta.encoder.layer:
    new_attn = ReplaceAttention(config=model.config, old_attn=layer.attention, rank=0)
    layer.attention.self = new_attn

with torch.no_grad():
    fixed_logits = model(**inputs).logits

print('Fixed logits:', fixed_logits.tolist())
max_diff = (orig_logits - fixed_logits).abs().max().item()
print(f'Max absolute logit difference: {max_diff:.8f}')
assert max_diff < 1e-4, f"Mismatch too high: {max_diff}"
print("VERIFICATION SUCCESSFUL! Logits match identically!")
