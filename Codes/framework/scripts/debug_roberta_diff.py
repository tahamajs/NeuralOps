import torch
from transformers import AutoModelForSequenceClassification
from framework.attentions.adaptive_span_attention import AdaptiveSpanAttention
from framework.core.attention_utils import inject_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("models/roberta-base").to(device)
ckpt = torch.load("results/unified_asvd/roberta/finetuned_model.pt", map_location=device)
model.load_state_dict(ckpt, strict=False)

layer0 = model.roberta.encoder.layer[0]
orig_self = layer0.attention.self

hidden_states = torch.randn(2, 128, 768, device=device)
with torch.no_grad():
    orig_out = orig_self(hidden_states)[0]

new_self = AdaptiveSpanAttention(model.config, pretrained_attn=orig_self, rank=32).to(device)
with torch.no_grad():
    new_out = new_self(hidden_states)[0]

diff = (orig_out - new_out).abs().max().item()
print("Single layer max diff between orig_self and new_self:", diff)
print("orig_out[:1, :2, :5]:\n", orig_out[:1, :2, :5])
print("new_out[:1, :2, :5]:\n", new_out[:1, :2, :5])

# Compare entire layer (including output.dense, LayerNorm)
with torch.no_grad():
    orig_layer_out = layer0(hidden_states)[0]

inject_attention(model, "adaptive_span", rank=32)
layer0_injected = model.roberta.encoder.layer[0]
with torch.no_grad():
    injected_layer_out = layer0_injected(hidden_states)[0]

diff_layer = (orig_layer_out - injected_layer_out).abs().max().item()
print("Entire Layer 0 max diff:", diff_layer)
