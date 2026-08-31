import torch
import traceback
import sys
sys.path.insert(0, "Codes")
from framework.eval_all_models_datasets import build_model_instance, apply_module_to_model

m, _ = build_model_instance('bert', num_labels=2, hidden_size=128, num_heads=4, num_layers=2, device=torch.device('cuda'))
try:
    apply_module_to_model(m, 'hybrid_svd', 'compressor', rank=16, device=torch.device('cuda'))
    print("HYBRID SVD SUCCESS")
except Exception as e:
    traceback.print_exc()

try:
    apply_module_to_model(m, 'direct_svd_encoder', 'compressor', rank=16, device=torch.device('cuda'))
    print("DIRECT SVD ENCODER SUCCESS")
except Exception as e:
    traceback.print_exc()

