import torch
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig
import sys
import os

# Ensure framework is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from framework.core.registry import COMPRESSOR_REGISTRY
from framework.eval_all_models_datasets import apply_module_to_model
import framework.compression_engine

def run_tests():
    compressors = list(COMPRESSOR_REGISTRY.keys())
    print(f"Found {len(compressors)} compressors in registry.")
    
    passed = 0
    failed = []
    
    for comp_name in compressors:
        try:
            print(f"Testing {comp_name}...")
            
            # 1. BERT
            config = BertConfig(hidden_size=128, num_attention_heads=4, num_hidden_layers=2)
            model = BertModel(config)
            
            model, _ = apply_module_to_model(model, comp_name, 'compressor', rank=16)
            
            # FP32 Forward + Backward
            model = model.train()
            dummy_input = torch.randint(0, 100, (2, 8))
            out = model(dummy_input)
            loss = out.last_hidden_state.sum()
            loss.backward()
            
            # FP16 Forward + Backward
            model = model.half()
            # Reset gradients
            model.zero_grad()
            out_fp16 = model(dummy_input)
            loss_fp16 = out_fp16.last_hidden_state.sum()
            loss_fp16.backward()
            
            # 2. RoBERTa
            config_roberta = RobertaConfig(hidden_size=128, num_attention_heads=4, num_hidden_layers=2)
            model_roberta = RobertaModel(config_roberta)
            model_roberta, _ = apply_module_to_model(model_roberta, comp_name, 'compressor', rank=16)
            
            passed += 1
            print(f"  [PASS] {comp_name}")
            
        except Exception as e:
            print(f"  [FAIL] {comp_name}: {str(e)}")
            failed.append((comp_name, str(e)))
            
    print("\n" + "="*40)
    print(f"Total compressors tested: {len(compressors)}")
    print(f"Total PASSED: {passed}")
    if failed:
        print("Failed compressors:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print("="*40)

if __name__ == "__main__":
    run_tests()
