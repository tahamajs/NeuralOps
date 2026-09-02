import torch
import sys
from transformers import AutoModelForSequenceClassification

print("=== STEP 2: MINIMAL REPRODUCER FOR JAMBA NAN ===")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load model in FP32
model_fp32 = AutoModelForSequenceClassification.from_pretrained('t5-small').to(device)
from framework.attentions.jamba import Jamba2Block
jamba_fp32 = Jamba2Block(config=model_fp32.config, d_model=512, n_heads=8).to(device)
model_fp32.transformer.encoder.block[0].layer[0].SelfAttention = jamba_fp32

# Create dummy inputs matching SST-2 WITH EOS TOKEN
input_ids = torch.randint(10, 1000, (2, 64), device=device)
input_ids[:, -1] = model_fp32.config.eos_token_id  # ensure EOS exists!
attention_mask = torch.ones((2, 64), device=device)
labels = torch.tensor([1, 0], device=device)

# --- Test 1: FP32 Forward ---
out_fp32 = model_fp32(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss_fp32 = out_fp32.loss
logits_fp32 = out_fp32.logits
print(f"[FP32 Forward] Loss: {loss_fp32.item():.4f} | Has NaN/Inf: {torch.isnan(loss_fp32).item() or torch.isinf(loss_fp32).item()}")
print(f"[FP32 Forward] Logits finite: {torch.isfinite(logits_fp32).all().item()}")

# --- Test 2: FP32 Backward ---
loss_fp32.backward()
has_nan_grad_fp32 = any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model_fp32.parameters())
print(f"[FP32 Backward] Any NaN/Inf gradient: {has_nan_grad_fp32}")

# --- Test 3: FP16 Forward ---
del model_fp32, jamba_fp32
torch.cuda.empty_cache()

model_fp16 = AutoModelForSequenceClassification.from_pretrained('t5-small').to(device).half()
jamba_fp16 = Jamba2Block(config=model_fp16.config, d_model=512, n_heads=8).to(device).half()
model_fp16.transformer.encoder.block[0].layer[0].SelfAttention = jamba_fp16

out_fp16 = model_fp16(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss_fp16 = out_fp16.loss
logits_fp16 = out_fp16.logits
print(f"\n[FP16 Forward] Loss: {loss_fp16.item()} | Has NaN/Inf: {torch.isnan(loss_fp16).item() or torch.isinf(loss_fp16).item()}")
print(f"[FP16 Forward] Logits finite: {torch.isfinite(logits_fp16).all().item()}")

# --- Test 4: FP16 Backward ---
loss_fp16.backward()
nan_params = [name for name, p in model_fp16.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all().item()]
print(f"[FP16 Backward] Any NaN/Inf gradient: {len(nan_params) > 0}")
if nan_params:
    print(f"[FP16 Backward] Parameters with NaN gradient (first 5): {nan_params[:5]}")
