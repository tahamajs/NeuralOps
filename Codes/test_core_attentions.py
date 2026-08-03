import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import BertConfig
from framework.core.registry import ATTENTION_REGISTRY
import framework.attentions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Testing core attention modules on device: {device}")

    config = BertConfig(
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    core_attns = ["full", "svd", "xattention", "linformer", "performer", "nystrom", "asvd", "gqa", "lsh"]
    
    for attn_name in core_attns:
        cls_or_fn = ATTENTION_REGISTRY.resolve(attn_name)
        module = cls_or_fn(config).to(device)
        module.train()

        x = torch.randn(2, 64, config.hidden_size, device=device, requires_grad=True)
        mask = torch.ones((2, 64), dtype=torch.long, device=device)
        mask[:, 32:] = 0

        out, _ = module(x, attention_mask=mask)
        loss = out.sum()
        loss.backward()

        nan_grads = False
        for p_name, p in module.named_parameters():
            if p.requires_grad and p.grad is not None:
                if torch.isnan(p.grad).any():
                    nan_grads = True
                    break

        print(f"Attention '{attn_name}': Output shape={out.shape} | NaN output={torch.isnan(out).any().item()} | NaN grads={nan_grads} -> ✅ PASSED")

if __name__ == "__main__":
    main()
