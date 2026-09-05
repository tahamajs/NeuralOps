import sys
import os
from pathlib import Path
import torch

if os.path.exists(r"D:\svd_project"):
    os.chdir(r"D:\svd_project")
    sys.path.insert(0, r"D:\svd_project\Codes")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.compression_engine.aa_svd_compressor import AASVDCompressor, SVDLinear
from framework.utils.lora import apply_lora

model_path = r"D:\svd_project\models\gpt2" if os.path.exists(r"D:\svd_project") else "gpt2"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tok = AutoTokenizer.from_pretrained(model_path)

# Test 1: Plain GPT-2
c = AASVDCompressor(model, tokenizer=tok, rank=288, max_error=0.50, target_attention_only=True)
plain_targets = [n for n, m in model.named_modules() if c._is_target_module(n, m)]
print(f"Plain GPT-2 target attention layers count: {len(plain_targets)}")
print("Sample plain targets:", plain_targets[:4])

# Test 2: GPT-2 with LoRA applied (as in sweep)
model_lora, _ = apply_lora(model, r=32)
c_lora = AASVDCompressor(model_lora, tokenizer=tok, rank=288, max_error=0.50, target_attention_only=True)
lora_targets = [n for n, m in model_lora.named_modules() if c_lora._is_target_module(n, m)]
print(f"LoRA GPT-2 target attention layers count: {len(lora_targets)}")
print("Sample LoRA targets:", lora_targets[:4])

# Test 3: Actual compression run on a single layer to verify shapes and factorization
target_name = lora_targets[0]
target_mod = dict(model_lora.named_modules())[target_name]
svd_layer, err, r = c_lora.compress_linear(target_mod, rank=288)
print(f"Compressed {target_name}: svd_layer={type(svd_layer).__name__}, err={err:.4f}, rank={r}")
if svd_layer is not None:
    print(f"  svd_layer.U shape: {svd_layer.U.shape}, svd_layer.V shape: {svd_layer.V.shape}")
    dummy_x = torch.randn(2, 16, 768)
    out = svd_layer(dummy_x)
    print(f"  Forward pass output shape: {out.shape} (expected [2, 16, 2304])")
