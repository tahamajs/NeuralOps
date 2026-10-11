import torch
import torch.nn as nn
import time
from neuralops.attentions.cosformer import CosformerAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, head_dim=32, d_ff=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CosformerAttention(hidden_size=d_model, num_heads=n_heads, head_dim=head_dim)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class ToyTransformer(nn.Module):
    def __init__(self, vocab_size=5000, d_model=256, num_layers=4, n_heads=8, head_dim=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, head_dim=head_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} ({torch.cuda.get_device_name(0)})')

    model = ToyTransformer(vocab_size=5000, d_model=256, num_layers=4, n_heads=8, head_dim=32).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model initialized: 4 layers, {total_params:,} trainable parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    batch_size = 8
    seq_len = 128
    vocab_size = 5000

    print('Starting 100-step training loop...')
    t0 = time.time()
    loss_history = []

    for step in range(1, 101):
        # Generate synthetic autoregressive language modeling batch
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step == 1 or step % 10 == 0:
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f'Step {step:3d}/100 | Loss: {loss_val:.4f} | Peak VRAM: {vram_mb:.1f} MB')

    elapsed = time.time() - t0
    print(f'Training completed in {elapsed:.2f}s ({100/elapsed:.1f} steps/s)')
    print(f'Initial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}')
    print('\nLoss Curve Summary (every 10 steps):')
    for s in range(0, 100, 10):
        bar = '=' * int(loss_history[s] * 3)
        print(f'Step {s+1:3d}: {loss_history[s]:.4f} | {bar}')

if __name__ == '__main__':
    main()
