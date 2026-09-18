import torch
import sys
from transformers import AutoModelForCausalLM

sys.path.append("D:/SVD_Project/Codes")
from framework.eval_all_models_datasets import apply_module_to_model

model = AutoModelForCausalLM.from_pretrained('D:/SVD_Project/models/gpt2').eval()
model, _ = apply_module_to_model(model, 'hla_inject', 'gpt')

with torch.no_grad():
    out = model(torch.randint(0, 1000, (1, 16)))
    print("Logits min/max:")
    print(out.logits.min(), out.logits.max())
    print("Has NaNs:", torch.isnan(out.logits).any().item())
