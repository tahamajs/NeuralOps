import sys
import os
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification

sys.path.insert(0, "Codes")
from framework.attentions.svd_llm_v2_attention import SVDLLMV2Attention, inject_svd_llm_v2
from framework.eval_all_models_datasets import apply_module_to_model

def test_qwen_and_energy():
    print("\n==================================================")
    print("Testing Qwen3.5 & Energy Threshold in SVDLLMV2")
    print("==================================================")
    class QwenMockConfig:
        hidden_size = 1024
        num_attention_heads = 8
        num_key_value_heads = 2
        head_dim = 256
        model_type = "qwen3_5"
        is_decoder = True
        
    cfg = QwenMockConfig()
    mod = SVDLLMV2Attention(config=cfg, energy_threshold=0.90, rank=64)
    print(f"Computed head_dim: {mod.head_dim} (expected 256)")
    print(f"Computed all_head_size: {mod.all_head_size} (expected 2048)")
    print(f"Computed kv_head_size: {mod.kv_head_size} (expected 512)")
    print(f"is_decoder: {mod.is_decoder} (expected True)")
    
    assert mod.head_dim == 256
    assert mod.all_head_size == 2048
    assert mod.kv_head_size == 512
    assert mod.is_decoder is True

    # Test energy threshold rank computation
    u = nn.Parameter(torch.empty(2048, 64))
    s = nn.Parameter(torch.empty(64))
    vh = nn.Parameter(torch.empty(64, 1024))
    b = nn.Parameter(torch.zeros(2048))
    
    # Create rank-10 matrix + small noise
    low_rank_W = torch.randn(2048, 10) @ torch.randn(10, 1024)
    mod._decompose_and_set(u, s, vh, b, low_rank_W, None)
    active_r = (s.abs() > 1e-4).sum().item()
    print(f"Energy threshold rank for rank-10 matrix: {active_r} (expected <= 11)")
    assert active_r <= 11, f"Rank {active_r} exceeds expected energy-based rank!"
    print("Qwen3.5 & Energy Threshold PASSED!")
    return True

def test_roberta_svd_llm_v2(device: str = "cpu"):
    print("\n==================================================")
    print(f"Testing svd_llm_v2 on RoBERTa (device={device})")
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
    
    model, stats = apply_module_to_model(model, "svd_llm_v2", "attention", rank=32, device=device)
    model = model.to(device)
    print(f"Injected svd_llm_v2 into RoBERTa successfully.")
    
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

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running SVDLLMV2 tests on device: {device}")
    
    ok1 = test_qwen_and_energy()
    ok2 = test_roberta_svd_llm_v2(device=device)
    
    if ok1 and ok2:
        print("\nALL SVDLLMV2Attention TESTS PASSED!")
    else:
        sys.exit(1)
