#!/usr/bin/env python3
import copy
import sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Codes"))

from framework.eval_all_models_datasets import apply_module_to_model

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
tok = AutoTokenizer.from_pretrained("models/gpt2_wikitext")
model = AutoModelForCausalLM.from_pretrained("models/gpt2_wikitext").cuda().half()
student, _ = apply_module_to_model(copy.deepcopy(model), "triton_svd_linear_attention", "attention", rank=64, device="cuda")
student.eval()

text = "\n\n".join(ds["text"])
enc = tok(text, return_tensors="pt")
ids = enc.input_ids[:, 768:1024].cuda()

with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        hidden_states = student.transformer.wte(ids) + student.transformer.wpe(torch.arange(256, device="cuda"))
        hidden_states = student.transformer.drop(hidden_states)
        
        for i in range(4):
            block = student.transformer.h[i]
            hidden_states = block(hidden_states)[0]
            
        block4 = student.transformer.h[4]
        norm1 = block4.ln_1(hidden_states)
        attn_mod = block4.attn._wrapped_attention
        
        B, N, C = norm1.shape
        q = attn_mod.q_proj(norm1).view(B, N, 12, 64).transpose(1, 2)
        k = attn_mod.k_proj(norm1).view(B, N, 12, 64).transpose(1, 2)
        v = attn_mod.v_proj(norm1).view(B, N, 12, 64).transpose(1, 2)
        
        print("Block 4 q shape:", q.shape, "dtype:", q.dtype)
        print("Block 4 k shape:", k.shape, "dtype:", k.dtype)
        print("Block 4 v shape:", v.shape, "dtype:", v.dtype)
        
        # Now trace fallback step by step
        import torch.nn.functional as F
        phi_q = F.elu(q.float()) + 1.0
        phi_k = F.elu(k.float()) + 1.0
        v_f = v.float()
        
        chunk_size = 64
        num_chunks = (N + chunk_size - 1) // chunk_size
        S = torch.zeros((B, 12, 64, 64), dtype=torch.float32, device="cuda")
        Z = torch.zeros((B, 12, 64), dtype=torch.float32, device="cuda")
        
        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)
            
            qc = phi_q[:, :, start:end]
            kc = phi_k[:, :, start:end]
            vc = v_f[:, :, start:end]
            
            num_inter = torch.matmul(qc, S)
            denom_inter = torch.matmul(qc, Z.unsqueeze(-1)).squeeze(-1)
            
            A_intra = torch.matmul(qc, kc.transpose(-1, -2))
            causal_mask = torch.tril(torch.ones(end - start, end - start, device="cuda", dtype=torch.bool))
            A_intra = torch.where(causal_mask, A_intra, torch.zeros_like(A_intra))
            num_intra = torch.matmul(A_intra, vc)
            denom_intra = A_intra.sum(dim=-1)
            
            denom = denom_inter.unsqueeze(-1) + denom_intra.unsqueeze(-1) + 1e-6
            num = num_inter + num_intra
            chunk_out = num / denom
            
            print(f"Chunk {c} [{start}:{end}]:")
            print(f"  num: min={num.min().item():.2f}, max={num.max().item():.2f}, nan={torch.isnan(num).any().item()}")
            print(f"  denom: min={denom.min().item():.4f}, max={denom.max().item():.2f}, nan={torch.isnan(denom).any().item()}")
            print(f"  chunk_out: nan={torch.isnan(chunk_out).any().item()}, inf={torch.isinf(chunk_out).any().item()}")
            
            S = S + torch.matmul(kc.transpose(-1, -2), vc)
            Z = Z + kc.sum(dim=-2)
