import torch
import traceback
import sys
sys.path.insert(0, "Codes")
from framework.eval_all_models_datasets import build_model_instance, apply_module_to_model

m, _ = build_model_instance('bert', num_labels=2, hidden_size=128, num_heads=4, num_layers=2, device=torch.device('cuda'))
try:
    apply_module_to_model(m, 'your_compressor_name', 'compressor', rank=16, device=torch.device('cuda'))
    print("YOUR COMPRESSOR SUCCESS")
except Exception as e:
    traceback.print_exc()

