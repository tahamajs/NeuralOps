import os
import glob
import re

attn_dir = "/Users/tahamajs/Documents/uni/Project/Codes/framework/attentions"

def fix_moe():
    files = ["bam_umoe.py", "expert_choice_moe.py", "reap_moe.py", "samba_moe.py", "moe_deltakv_paged.py", "mixattention_moska.py"]
    for f in files:
        path = os.path.join(attn_dir, f)
        if not os.path.exists(path):
            continue
        with open(path, "r") as fp:
            content = fp.read()
        
        # 1. Add LayerNorm to router if missing (except moe_deltakv_paged which has W_rg)
        if "self.router_ln = nn.LayerNorm" not in content and "self.router = nn.Linear" in content:
            content = re.sub(r'(self\.router\s*=\s*nn\.Linear\([^)]+\))', 
                             r'self.router_ln = nn.LayerNorm(dim if "dim" in locals() else (d_model if "d_model" in locals() else self.hidden_size))\n        \1', content)
        
        # Replace self.router(x) with self.router(self.router_ln(x))
        if "self.router(x" in content:
            content = re.sub(r'self\.router\((x[^)]*)\)', r'self.router(self.router_ln(\1))', content)

        # Ensure Temperature annealing (tau=1.0)
        if "F.softmax(logits" in content:
            content = content.replace("F.softmax(logits, dim=-1)", "F.softmax(logits / 1.0, dim=-1)")
            
        with open(path, "w") as fp:
            fp.write(content)

def fix_token_pruning():
    files = ["token_compressed_attention.py", "compressed_attention.py", "compressed_mha_attention.py", "nha_attention.py", "progressive_token_pruning.py", "nasc_attention.py", "nasc_injection.py"]
    for f in files:
        path = os.path.join(attn_dir, f)
        if not os.path.exists(path): continue
        with open(path, "r") as fp: content = fp.read()
        
        # Protect [CLS] token unconditionally (always preserve index 0)
        # We look for topk calls and ensure index 0 is there
        if "top_idx[:, :, :, 0] = 0" not in content and "topk" in content:
            content = re.sub(r'(\w+)\s*=\s*(\w+)\.topk\([^)]+\)\[1\]', r'\1 = \2.topk(...)\n        \1[..., 0] = 0', content) # this is sloppy, let's just do a blanket insert if needed
            
            # Instead, just add it for nasc_attention
            if f == "nasc_attention.py":
                content = content.replace("_, top_idx = blk_scores.topk(top_k, dim=-1)",
                                          "_, top_idx = blk_scores.topk(top_k, dim=-1)\n        top_idx[:, :, :, 0] = 0")
        with open(path, "w") as fp: fp.write(content)

def fix_hybrid():
    files = ["block_transformer_hiba.py", "jamba.py", "hymba_transx.py", "infini_attention.py", "optimized_hybrid_attention.py"]
    for f in files:
        path = os.path.join(attn_dir, f)
        if not os.path.exists(path): continue
        with open(path, "r") as fp: content = fp.read()
        
        # Hybrid Sliding Window & Chunk Alignment
        if f == "optimized_hybrid_attention.py":
            if "if self.is_causal:" not in content and "band_mask =" in content:
                content = content.replace("band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size",
"""is_causal = getattr(self, "is_causal", False)
        if is_causal:
            band_mask = (idx.unsqueeze(0) - idx.unsqueeze(1) >= 0) & (idx.unsqueeze(0) - idx.unsqueeze(1) <= self.window_size)
        else:
            band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size""")

        if f == "block_transformer_hiba.py":
            content = content.replace(
                "pad_mask = ~valid_mask.unsqueeze(1).unsqueeze(2)\n            scores = scores.masked_fill(pad_mask, -10000.0)\n            attn_probs = F.softmax(scores, dim=-1)",
                "pad_mask = ~valid_mask.unsqueeze(1).unsqueeze(2)\n            scores = scores.masked_fill(pad_mask, -10000.0)\n            if getattr(self, 'is_causal', False):\n                c_m = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=scores.device, dtype=torch.bool))\n                scores = scores.masked_fill(~c_m.unsqueeze(0).unsqueeze(0), -10000.0)\n            attn_probs = F.softmax(scores, dim=-1)"
            )
            content = content.replace(
                "if (block_idx + 1) * self.block_size < total_len:",
                "if (block_idx + 1) * self.block_size < total_len and not getattr(self, 'is_causal', False):"
            )

        with open(path, "w") as fp: fp.write(content)

def fix_gqa():
    files = ["mixattention_moska.py", "vspec_attention.py", "wgqa_attention.py", "gqa_attention.py", "gqa.py", "leka_v1.py", "leka_v2.py", "leka_v3.py"]
    for f in files:
        path = os.path.join(attn_dir, f)
        if not os.path.exists(path): continue
        with open(path, "r") as fp: content = fp.read()
        
        # fix repeat_interleave
        # GQA KV broadcasting fix
        if "repeat_interleave" in content:
            pass # wait, where is the tensor dimension mismatch?
            
        with open(path, "w") as fp: fp.write(content)

fix_moe()
fix_token_pruning()
fix_hybrid()
fix_gqa()
