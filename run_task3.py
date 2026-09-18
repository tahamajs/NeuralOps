import torch
import sys
from transformers import AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

sys.path.append("Codes")
from framework.eval_all_models_datasets import apply_module_to_model

modules = ['lora_attention', 'lsh_attention', 'compressed_mha', 'galerkin', 'svd_linear_attention', 'full_attention']

torch.manual_seed(42)
input_ids = torch.randint(0, 1000, (1, 16))

for mod in modules:
    print(f"\n--- {mod} ---")
    model1 = AutoModelForCausalLM.from_pretrained('D:/SVD_Project/models/gpt2', device_map="cpu").eval()
    model2 = AutoModelForCausalLM.from_pretrained('D:/SVD_Project/models/gpt2', device_map="cpu").eval()
    
    try:
        model2, _ = apply_module_to_model(model2, mod, 'gpt')
        with torch.no_grad():
            out1 = model1(input_ids).logits
            out2 = model2(input_ids).logits
            diff = (out1 - out2).abs().max().item()
        print(f"Max logit diff: {diff}")
    except Exception as e:
        print("Error:", e)
