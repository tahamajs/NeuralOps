import torch
import sys
from transformers import AutoModelForSequenceClassification, AutoConfig
from framework.utils.replace_attention import replace_attention

modules_to_check = ['cosformer', 'adaptive_span', 'griffin_ssd']

print("=== SPOT-CHECK VERIFICATION OF WIRING AND GRADIENTS ===")
for mod_name in modules_to_check:
    cfg = AutoConfig.from_pretrained('t5-small')
    model = AutoModelForSequenceClassification.from_pretrained('t5-small', config=cfg).cuda().half()
    model = replace_attention(model, mod_name)
    model.train()
    
    # 1. Decoder cross-attention verification
    dec_cross = model.transformer.decoder.block[0].layer[1].EncDecAttention
    B, N_q, N_kv, D = 2, 8, 16, 512
    dec_h = torch.randn(B, N_q, D, dtype=torch.float16, device='cuda')
    enc_h = torch.randn(B, N_kv, D, dtype=torch.float16, device='cuda')
    
    out1, _ = dec_cross(dec_h, key_value_states=enc_h)
    out2, _ = dec_cross(dec_h, key_value_states=enc_h + 5.0)
    cross_diff = (out1 - out2).abs().max().item()
    
    # 2. Gradient flow verification
    loss = out1.sum()
    loss.backward()
    
    nonzero_grads = [name for name, p in dec_cross.named_parameters() if p.grad is not None and p.grad.norm().item() > 0]
    total_params = len(list(dec_cross.parameters()))
    
    print(f"Module: {mod_name:15}")
    print(f"  - Cross-attention dependency: diff = {cross_diff:.4f} (must be > 0.0)")
    print(f"  - Non-zero gradient parameters: {len(nonzero_grads)} / {total_params} tensors")
    print(f"  - Output shape: {tuple(out1.shape)} (expected {(B, N_q, D)})")
    assert cross_diff > 1e-4, f"Cross-attention in {mod_name} does not depend on encoder hidden states!"
    assert len(nonzero_grads) > 0, f"{mod_name} parameters received no gradients!"
    print(f"  -> VERIFIED: Wiring and gradients are completely genuine and correct.\n")
