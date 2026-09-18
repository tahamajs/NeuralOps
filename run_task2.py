import torch
import sys
from transformers import AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

sys.path.append("Codes")
from framework.eval_all_models_datasets import apply_module_to_model

modules = ['lora_attention', 'lsh_attention', 'compressed_mha', 'galerkin', 'svd_linear_attention', 'full_attention']

for mod in modules:
    print(f"\n--- {mod} ---")
    model = AutoModelForCausalLM.from_pretrained('D:/SVD_Project/models/gpt2', device_map="cpu").eval()
    before_keys = set(model.state_dict().keys())
    before_sum = sum(p.sum().item() for p in model.parameters())
    
    try:
        model, _ = apply_module_to_model(model, mod, 'gpt')
        after_keys = set(model.state_dict().keys())
        after_sum = sum(p.sum().item() for p in model.parameters())
        
        diff = after_sum - before_sum
        new_keys = after_keys - before_keys
        
        print(f"New keys added? {'YES' if len(new_keys) > 0 else 'NO'}")
        print(f"Number of new keys: {len(new_keys)}")
        print(f"Parameter delta: {diff}")
    except Exception as e:
        print("Error:", e)
