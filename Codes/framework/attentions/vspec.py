#!/usr/bin/env python3
"""
VSpec++: Variational Spectral Token-Adaptive Attention
Production script for DistilBERT with spectral, head, and token-adaptive compression.

Usage:
    python VSpecAttention.py --task sst2 --epochs 5 --batch_size 16 --lr 1e-5
"""

import os, argparse, logging, warnings
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import svd

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, set_seed,
)
from datasets import Dataset, DatasetDict

# ------- Offline environment and logging --------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ================================================================================
# 1. Spectral utilities & Lagrangian rank allocation (unchanged)
# ================================================================================

def calculate_spectral_entropy(singular_values: np.ndarray) -> float:
    s = singular_values.astype(np.float64)
    s_sum = s.sum()
    if s_sum == 0:
        return 0.0
    prob = s / s_sum
    prob = prob[prob > 1e-12]
    return float(-np.sum(prob * np.log(prob)))

def compute_cumulative_entropy(singular_values: np.ndarray, max_rank: int) -> np.ndarray:
    s = singular_values.astype(np.float64)
    s_sq = s ** 2
    total = s_sq.sum()
    if total == 0:
        return np.ones(max_rank)
    cum = np.cumsum(s_sq[:max_rank]) / total
    return cum

def lagrangian_rank_allocation(
    weight_matrices: List[np.ndarray],
    target_budget: int,
    layer_proj_names: List[str],
    max_rank_factor: float = 0.5,
    lambda_bracket: Tuple[float,float] = (1e-8, 10.0),
    num_lambda_steps: int = 100,
) -> Dict[str, int]:
    max_ranks = []
    cost_table = []
    param_table = []
    for W, name in zip(weight_matrices, layer_proj_names):
        d_in, d_out = W.shape[1], W.shape[0]
        max_r = min(d_in, d_out, int(max_rank_factor * min(d_in, d_out)))
        max_r = max(1, max_r)
        max_ranks.append(max_r)
        _, s, _ = svd(W.astype(np.float64), full_matrices=False)
        cum_ent = compute_cumulative_entropy(s, max_r)
        cost = 1.0 - cum_ent
        cost_table.append(cost)
        params_per_r = np.arange(1, max_r + 1) * (d_in + d_out)
        param_table.append(params_per_r)
    low, high = lambda_bracket
    best_allocation = None
    best_used_params = None
    for _ in range(num_lambda_steps):
        lam = (low + high) / 2.0
        ranks = []
        total_params = 0
        for i in range(len(weight_matrices)):
            obj = cost_table[i] + lam * param_table[i]
            best_r_idx = int(np.argmin(obj))
            r = best_r_idx + 1
            ranks.append(r)
            total_params += param_table[i][best_r_idx]
        if total_params <= target_budget:
            best_allocation = ranks.copy()
            best_used_params = total_params
            high = lam
        else:
            low = lam
    if best_allocation is None:
        best_allocation = [1] * len(weight_matrices)
        log.warning("Could not meet budget, falling back to rank=1 everywhere.")
    rank_dict = {name: r for name, r in zip(layer_proj_names, best_allocation)}
    return rank_dict

# ================================================================================
# 2. VSpec++ PyTorch Attention Layer
# ================================================================================

class VariationalSpectralGate(nn.Module):
    """Rank gates: per-layer, outputs (B, rank)"""
    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.rank = rank
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, rank*2)
        )
    def forward(self, x):
        pooled = x.mean(dim=1)
        stats = self.net(pooled)    # (B, rank*2)
        mu, log_sigma = stats.chunk(2, dim=-1)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(log_sigma)
        g = torch.sigmoid(z)
        return g, mu, log_sigma

class HeadGate(nn.Module):
    """Head gates: outputs (B, n_heads)"""
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_heads)
        )
    def forward(self, x):
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.net(pooled))

class TokenRouter(nn.Module):
    """Token routing: top-k per batch, outputs (B, K, hidden), indices"""
    def __init__(self, hidden_dim, keep_ratio=0.5, min_tokens=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
    def forward(self, x):
        B, N, D = x.shape
        scores = torch.sigmoid(self.net(x).squeeze(-1))
        keep = max(self.min_tokens, int(N * self.keep_ratio))
        idx = scores.topk(keep, dim=1).indices
        routed = torch.stack([x[b, idx[b]] for b in range(B)])
        return routed, idx

class DynamicLowRankProjector(nn.Module):
    """Low-rank projector with dynamic gates"""
    def __init__(self, W, rank):
        super().__init__()
        U,S,Vh = torch.linalg.svd(W, full_matrices=False)
        self.U = nn.Parameter(U[:, :rank], requires_grad=False)
        self.S = nn.Parameter(S[:rank], requires_grad=False)
        self.Vh = nn.Parameter(Vh[:rank, :], requires_grad=False)
        self.rank = rank
    def forward(self, x, gate):
        # x: [B, N, D]
        S_eff = self.S.unsqueeze(0) * gate    # (B, rank)
        mid = torch.matmul(x, self.Vh.t())    # (B,N,rank)
        mid = mid * S_eff.unsqueeze(1)        # (B,N,rank)
        out = torch.matmul(mid, self.U.t())   # (B,N,D_out)
        return out

class VSpecAttention(nn.Module):
    """
    Replaces DistilBERT attention layer with VSpec++:
      - Dynamic low-rank projections Q/K/V/O (per layer, per batch)
      - Spectral gates
      - Head gates
      - Token router
    """
    def __init__(self, hidden_size, num_heads, rank, pretrained_attn):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rank = rank
        # Projector weights (factorized from original)
        self.Pq = DynamicLowRankProjector(
            pretrained_attn.q_lin.weight.data.clone().t(), rank)
        self.Pk = DynamicLowRankProjector(
            pretrained_attn.k_lin.weight.data.clone().t(), rank)
        self.Pv = DynamicLowRankProjector(
            pretrained_attn.v_lin.weight.data.clone().t(), rank)
        self.Po = DynamicLowRankProjector(
            pretrained_attn.out_lin.weight.data.clone(), rank)
        # Gates
        self.sg_q = VariationalSpectralGate(hidden_size, rank)
        self.sg_k = VariationalSpectralGate(hidden_size, rank)
        self.sg_v = VariationalSpectralGate(hidden_size, rank)
        self.head_gate = HeadGate(hidden_size, num_heads)
        self.router = TokenRouter(hidden_size)

        self.dropout = nn.Dropout(0.1)      # keep dropout

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape
        # Token router
        routed, idx = self.router(hidden_states)
        # Spectral gates for each projection
        gq, _, _ = self.sg_q(hidden_states)
        gk, _, _ = self.sg_k(hidden_states)
        gv, _, _ = self.sg_v(hidden_states)
        # Low-rank projections
        Q = self.Pq(routed, gq)
        K = self.Pk(routed, gk)
        V = self.Pv(routed, gv)
        # Reshape heads
        Q = Q.view(B, Q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, K.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, V.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        # Head gates
        gates = self.head_gate(hidden_states).unsqueeze(-1).unsqueeze(-1)
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:,None,None,:]
            else:
                mask = attention_mask
            scores = scores + mask
        scores = scores * gates
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1,2).contiguous().view(B, Q.shape[1], self.hidden_size)
        # Output projection
        out = self.Po(context, gq)
        # Reverse routing to original token positions
        output = hidden_states.clone()
        for b in range(B):
            output[b, idx[b]] = out[b]
        if output_attentions:
            return output, attn_weights
        return output, None

# ================================================================================
# 3. Injection helpers
# ================================================================================

def gather_all_attention_weights(model) -> Tuple[list, list]:
    weights = []
    names = []
    for layer_idx in range(6):
        distil_layer = model.distilbert.transformer.layer[layer_idx]
        attn = distil_layer.attention
        for proj in ['q_lin', 'k_lin', 'v_lin', 'out_lin']:
            w = getattr(attn, proj).weight.data.cpu().numpy()
            proj_short = proj.replace('_lin','')
            names.append(f"layer{layer_idx}_{proj_short}")
            weights.append(w)
    return weights, names

def inject_all_layers_vspec(model, rank_dict: Dict[str, int], config):
    for layer_idx in range(6):
        distil_layer = model.distilbert.transformer.layer[layer_idx]
        old_attn = distil_layer.attention
        for proj in ['q', 'k', 'v', 'out']:
            key = f"layer{layer_idx}_{proj}"
            if proj == 'out':
                ranks = {'o': rank_dict[key], 'q': rank_dict[f"layer{layer_idx}_q"],
                         'k': rank_dict[f"layer{layer_idx}_k"],
                         'v': rank_dict[f"layer{layer_idx}_v"]}
            else:
                ranks = {'q': rank_dict[f"layer{layer_idx}_q"], 'k': rank_dict[f"layer{layer_idx}_k"],
                         'v': rank_dict[f"layer{layer_idx}_v"], 'o': rank_dict[f"layer{layer_idx}_out"]}
        new_attn = VSpecAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            rank=rank_dict[f"layer{layer_idx}_q"],
            pretrained_attn=old_attn,
        )
        distil_layer.attention = new_attn

# ================================================================================
# 4. GLUE data loading (unchanged)
# ================================================================================

GLUE_TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def load_glue_local(task_name: str) -> DatasetDict:
    from framework.utils.paths import get_glue_dir
    base_path = str(get_glue_dir())
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Dataset not found: {task_path}")
    splits = {}
    for fname in os.listdir(task_path):
        if fname.endswith(".parquet"):
            split = fname.split("-")[0]
            splits[split] = Dataset.from_parquet(os.path.join(task_path, fname))
    if task_name == "mnli":
        for extra in ["validation_matched", "validation_mismatched"]:
            fpath = os.path.join(task_path, f"{extra}.parquet")
            if os.path.exists(fpath):
                splits[extra] = Dataset.from_parquet(fpath)
    return DatasetDict(splits)

def get_num_labels_and_is_regression(task: str):
    if task == "stsb":
        return 1, True
    if task == "cola":
        return 2, False
    mapping = {"sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3,
               "qnli": 2, "rte": 2, "wnli": 2}
    return mapping[task], False

def preprocess_function(examples, tokenizer, sent1_key, sent2_key,
                        max_length, is_regression):
    if sent2_key is None:
        t = tokenizer(examples[sent1_key], truncation=True,
                      padding="max_length", max_length=max_length)
    else:
        t = tokenizer(examples[sent1_key], examples[sent2_key],
                      truncation=True, padding="max_length", max_length=max_length)
    t["labels"] = [float(l) for l in examples["label"]] if is_regression else examples["label"]
    return t

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": (preds == labels).mean().item()}

# ================================================================================
# 5. Main Training Script
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train DistilBERT with VSpec++ Attention"
    )
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./vspec_output")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budget_ratio",
        type=float,
        default=0.2,
        help="Target fraction of original attention parameters (e.g. 0.2 = 20%%)"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Data -----
    raw_datasets = load_glue_local(args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                              local_files_only=True)
    num_labels, is_regression = get_num_labels_and_is_regression(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    tokenized = raw_datasets.map(
        lambda ex: preprocess_function(ex, tokenizer, sent1_key, sent2_key,
                                       args.max_len, is_regression),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    train_dataset = tokenized["train"]
    eval_dataset = tokenized.get("validation", tokenized.get("validation_matched"))

    log.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # ----- Model & rank optimisation -----
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=config, local_files_only=True)

    all_weights, all_names = gather_all_attention_weights(model)
    original_total_params = sum(w.size for w in all_weights)
    target_budget = int(original_total_params * args.budget_ratio)
    log.info(f"Original attention parameters: {original_total_params:,}, "
             f"target budget ({args.budget_ratio*100:.0f}%): {target_budget:,}")

    rank_dict = lagrangian_rank_allocation(
        weight_matrices=all_weights,
        target_budget=target_budget,
        layer_proj_names=all_names,
    )
    avg_rank = np.mean(list(rank_dict.values()))
    log.info(f"Optimised ranks across layers: average rank = {avg_rank:.1f}")

    inject_all_layers_vspec(model, rank_dict, config)
    log.info("Injected VSpec++ Attention.")

    # ----- Training -----
    total_steps = (len(train_dataset) // args.batch_size) * args.epochs
    warmup_steps = int(0.06 * total_steps)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    log.info("Starting training with VSpec++ Attention...")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, args.task))
    log.info("Training completed.")

if __name__ == "__main__":
    main()
