import sys
import os
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

sys.path.insert(0, "Codes")
from framework.eval_all_models_datasets import apply_module_to_model

def test_module_on_distilbert(module_name: str, device: str = "cpu"):
    print(f"\n==================================================")
    print(f"Testing {module_name} on DistilBERT (device={device})")
    print(f"==================================================")
    
    config = DistilBertConfig(
        vocab_size=1000,
        dim=768,
        n_heads=12,
        n_layers=2,
        hidden_dim=1024,
        num_labels=2,
    )
    model = DistilBertForSequenceClassification(config)
    model.to(device)
    
    # Apply module
    model, stats = apply_module_to_model(model, module_name, "attention", rank=32, device=device)
    model = model.to(device)
    print(f"Module {module_name} injected successfully.")
    
    # Verify module parameters device
    mismatched = []
    for name, param in model.named_parameters():
        if param.device.type != torch.device(device).type:
            mismatched.append((name, str(param.device)))
    if mismatched:
        print(f"ERROR: Found parameters on wrong device: {mismatched[:5]}")
        return False
        
    # Forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        print(f"Forward pass SUCCESS! Loss: {loss.item():.4f}, Logits shape: {logits.shape}")
        
        loss.backward()
        optimizer.step()
        print(f"Backward pass and optimizer step SUCCESS!")
        return True
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running tests on device: {device}")
    
    modules = ["nexus", "minimal_nova", "svd_compressed"]
    results = {}
    for mod in modules:
        res = test_module_on_distilbert(mod, device=device)
        results[mod] = res
        
    print("\n==================================================")
    print("SUMMARY RESULTS ON DISTILBERT:")
    for mod, res in results.items():
        print(f"  {mod}: {'PASS' if res else 'FAIL'}")
    print("==================================================")
    
    if not all(results.values()):
        sys.exit(1)
