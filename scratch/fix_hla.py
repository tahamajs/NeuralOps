# Add 1/sqrt(d) scaling to hla_attention to prevent exploding logits
import re
with open("Codes/framework/attentions/hla_attention.py", "r") as f:
    text = f.read()

# Fix the missing scaling factor
text = text.replace(
    "qk = torch.matmul(q_f, k_f.transpose(-1, -2))  # [B, H, L, L]",
    "qk = torch.matmul(q_f, k_f.transpose(-1, -2)) / math.sqrt(d) # [B, H, L, L]"
)
with open("Codes/framework/attentions/hla_attention.py", "w") as f:
    f.write(text)
