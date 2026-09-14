import sys
import os
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

sys.path.insert(0, "Codes")
from framework.eval_all_models_datasets import apply_module_to_model

def test_module(name: str, device: str = "cpu"):
    config = RobertaConfig(
        vocab_size=1000,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=2,
        intermediate_size=1024,
        max_position_embeddings=512,
        num_labels=2,
    )
    model = RobertaForSequenceClassification(config)
    model.to(device)
    
    try:
        model, stats = apply_module_to_model(model, name, "attention", rank=32, device=device)
        model = model.to(device)
    except Exception as e:
        return False, f"Injection failed: {e}"
        
    input_ids = torch.randint(0, 1000, (2, 16), device=device)
    attention_mask = torch.ones((2, 16), device=device)
    labels = torch.randint(0, 2, (2,), device=device)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        return True, f"Loss: {loss.item():.4f}"
    except Exception as e:
        return False, f"Forward/Backward failed: {e}"

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing batch on RoBERTa ({device})...\n")
    
    test_list = [
        "svd_compressed",
        "nexus",
        "minimal_nova",
        "minimal_prism",
        "svd_llm_v2",
        "moba_attention",
        "orion",
        "gqa_attention",
    ]
    
    all_ok = True
    for mod in test_list:
        ok, msg = test_module(mod, device=device)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {mod:25s} -> {msg}")
        if not ok:
            all_ok = False
            
    print("\n" + "="*50)
    if all_ok:
        print("ALL TESTED MODULES PASSED ON ROBERTA!")
    else:
        print("SOME MODULES FAILED!")
        sys.exit(1)
