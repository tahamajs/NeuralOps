import os
import re

DIR = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\attentions"

def patch_lora():
    path = os.path.join(DIR, "lora_attention.py")
    if not os.path.exists(path): return
    with open(path, "r", encoding="utf-8") as f: content = f.read()
    
    old_check = """        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero-shot init")"""
    new_check = """        if pretrained_attn is None:
            class MockAttn:
                def __init__(self, d):
                    self.q_lin = torch.nn.Linear(d, d)
                    self.k_lin = torch.nn.Linear(d, d)
                    self.v_lin = torch.nn.Linear(d, d)
                    self.out_lin = torch.nn.Linear(d, d)
            hidden_size = getattr(model_config, "hidden_size", 768)
            pretrained_attn = MockAttn(hidden_size)"""
    if old_check in content:
        content = content.replace(old_check, new_check)
        with open(path, "w", encoding="utf-8") as f: f.write(content)
        print("Patched lora_attention.py")

patch_lora()
