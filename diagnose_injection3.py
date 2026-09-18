import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("D:/SVD_Project/Codes")
from framework.eval_all_models_datasets import apply_module_to_model

def get_checksum(state_dict):
    return sum(t.float().sum().item() for t in state_dict.values())

tokenizer = AutoTokenizer.from_pretrained("D:/SVD_Project/models/gpt2")
text = "The quick brown fox"
inputs = tokenizer(text, return_tensors="pt")

modules_to_test = ["svd_linear_attention", "gqa", "galerkin", "full_attention"]

base_model = AutoModelForCausalLM.from_pretrained("D:/SVD_Project/models/gpt2")
base_model.eval()
base_checksum = get_checksum(base_model.state_dict())
with torch.no_grad():
    base_logits = base_model(**inputs).logits

for mod in modules_to_test:
    print(f"\n=== MODULE: {mod} ===")
    model = AutoModelForCausalLM.from_pretrained("D:/SVD_Project/models/gpt2")
    model.eval()
    model, _ = apply_module_to_model(model, mod, "gpt")
    model.eval()
    mod_checksum = get_checksum(model.state_dict())
    with torch.no_grad():
        mod_logits = model(**inputs).logits
    
    diff = torch.max(torch.abs(base_logits - mod_logits)).item()
    print(f"Checksum changed? {base_checksum != mod_checksum} (Diff: {mod_checksum - base_checksum})")
    print(f"Max logit diff: {diff}")
