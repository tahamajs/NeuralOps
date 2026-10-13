import torch
from transformers import GPT2LMHeadModel
from neuralops.attentions.linear.gla_attention import GLAttention


def inject_gla(model):
    for block in model.transformer.h:
        old = block.attn
        new = GLAttention(
            hidden_size=model.config.n_embd,
            num_heads=model.config.n_head,
            head_dim=model.config.n_embd // model.config.n_head,
            old_attn=old,
        ).to(model.device)
        block.attn = new
    return model


if __name__ == "__main__":
    device = "cuda"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model = inject_gla(model)

    input_ids = torch.randint(0, 50257, (2, 128), device=device)
    with torch.no_grad():
        out = model(input_ids)
    print(f"✅ GPT-2 + GLA forward OK: {out.logits.shape}")

    # Count GLA layers
    n_gla = sum(1 for m in model.modules() if isinstance(m, GLAttention))
    print(f"   GLA layers: {n_gla}")
    print(f"   FLA enabled: {model.transformer.h[0].attn._fla_kernel is not None}")
