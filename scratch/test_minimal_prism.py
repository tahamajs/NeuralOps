import sys
import os
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification

sys.path.insert(0, "Codes")
from framework.attentions.minimal_prism import MinimalPRISMAttention, inject_minimal_prism
from framework.eval_all_models_datasets import apply_module_to_model

def test_qwen_dimensions():
    print("\n==================================================")
    print("Testing Qwen3.5 Dimension Inference")
    print("==================================================")
    # Qwen3.5-0.8B config mock
    class QwenMockConfig:
        hidden_size = 1024
        num_attention_heads = 8
        num_key_value_heads = 2
        head_dim = 256
        model_type = "qwen3_5"
        is_decoder = True
        
    cfg = QwenMockConfig()
    mod = MinimalPRISMAttention(model_config=cfg)
    print(f"Computed head_dim: {mod.head_dim} (expected 256)")
    print(f"Computed all_head_size: {mod.all_head_size} (expected 2048)")
    print(f"Computed kv_head_size: {mod.kv_head_size} (expected 512)")
    print(f"is_decoder: {mod.is_decoder} (expected True)")
    
    assert mod.head_dim == 256, f"head_dim mismatch: {mod.head_dim}"
    assert mod.all_head_size == 2048, f"all_head_size mismatch: {mod.all_head_size}"
    assert mod.kv_head_size == 512, f"kv_head_size mismatch: {mod.kv_head_size}"
    assert mod.is_decoder is True, f"is_decoder mismatch: {mod.is_decoder}"
    print("Qwen3.5 dimension check PASSED!")
    return True

def test_roberta_prism(device: str = "cpu"):
    print("\n==================================================")
    print(f"Testing minimal_prism on RoBERTa (device={device})")
    print("==================================================")
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
    
    model, stats = apply_module_to_model(model, "minimal_prism", "attention", rank=32, device=device)
    model = model.to(device)
    print(f"Injected minimal_prism into RoBERTa successfully.")
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    print(f"RoBERTa Forward pass SUCCESS! Loss: {loss.item():.4f}, Logits: {logits.shape}")
    
    loss.backward()
    optimizer.step()
    print("RoBERTa Backward pass SUCCESS!")
    return True

def test_distilbert_prism(device: str = "cpu"):
    print("\n==================================================")
    print(f"Testing minimal_prism on DistilBERT (device={device})")
    print("==================================================")
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
    
    model, stats = apply_module_to_model(model, "minimal_prism", "attention", rank=32, device=device)
    model = model.to(device)
    print(f"Injected minimal_prism into DistilBERT successfully.")
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    print(f"DistilBERT Forward pass SUCCESS! Loss: {loss.item():.4f}, Logits: {logits.shape}")
    
    loss.backward()
    optimizer.step()
    print("DistilBERT Backward pass SUCCESS!")
    return True

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running verification on device: {device}")
    
    ok1 = test_qwen_dimensions()
    ok2 = test_roberta_prism(device=device)
    ok3 = test_distilbert_prism(device=device)
    
    if ok1 and ok2 and ok3:
        print("\nALL MinimalPRISMAttention TESTS PASSED!")
    else:
        sys.exit(1)
