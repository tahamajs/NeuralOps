"""
Training script for Mamba vs. Transformer from scratch on TinyStories.

Usage:
    python train_mamba_scratch.py --model-type mamba --output_dir results/mamba_scratch
    python train_mamba_scratch.py --model-type transformer --output_dir results/transformer_scratch
"""

from __future__ import annotations
import argparse
import logging
import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from framework.attentions.mamba_block import MambaBlock

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, is_causal=(causal_mask is None))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 256, n_layers: int = 6, n_heads: int = 8, d_ff: int = 1024, max_seq_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=input_ids.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)
        x = self.ln_f(x)
        return self.lm_head(x)


class MambaLMBlock(nn.Module):
    def __init__(self, d_model: int = 256, d_state: int = 16, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            is_decoder_override=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        out, _ = self.mamba(x_norm)
        return residual + out


class MambaLM(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 256, n_layers: int = 6, d_state: int = 16, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaLMBlock(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Synthetic / Tokenized Streaming Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def load_tinystories_data(seq_len: int = 512) -> Tuple[Dataset, Dataset]:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        ds = load_dataset("roneneldan/TinyStories")
        
        train_texts = ds["train"]["text"][:20000]
        val_texts = ds["validation"]["text"][:2000]

        train_enc = tokenizer(" ".join(train_texts), return_tensors="pt")["input_ids"][0]
        val_enc = tokenizer(" ".join(val_texts), return_tensors="pt")["input_ids"][0]
        return TokenDataset(train_enc, seq_len), TokenDataset(val_enc, seq_len)
    except Exception as e:
        logger.warning("Could not load TinyStories from HF (%s). Using synthetic dataset fallback.", e)
        synth_train = torch.randint(0, 50257, (seq_len * 200 + 1,))
        synth_val = torch.randint(0, 50257, (seq_len * 20 + 1,))
        return TokenDataset(synth_train, seq_len), TokenDataset(synth_val, seq_len)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            count += 1
    model.train()
    avg_loss = total_loss / max(1, count)
    return math.exp(min(20.0, avg_loss))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=["mamba", "transformer"], default="mamba")
    parser.add_argument("--output_dir", type=str, default="results/mamba_scratch")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_type == "mamba":
        model = MambaLM(vocab_size=50257, d_model=256, n_layers=6, d_state=16, expand=2, d_conv=4)
    else:
        model = TransformerLM(vocab_size=50257, d_model=256, n_layers=6, n_heads=8, d_ff=1024, max_seq_len=args.seq_len)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Initialized {args.model_type.upper()} with {total_params:,} parameters.")
    model = model.to(device)

    train_ds, val_ds = load_tinystories_data(args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.total_steps,
        pct_start=args.warmup_steps / max(1, args.total_steps),
    )

    step = 0
    epoch = 0
    model.train()
    logger.info(f"Starting training for {args.total_steps} steps...")

    while step < args.total_steps:
        epoch += 1
        for x, y in train_loader:
            if step >= args.total_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1

            if step % args.eval_every == 0 or step == 1:
                ppl = evaluate(model, val_loader, device)
                logger.info(f"Step {step}/{args.total_steps} | Loss: {loss.item():.4f} | Val PPL: {ppl:.2f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    logger.info("Training complete and model checkpoint saved.")


if __name__ == "__main__":
    main()
