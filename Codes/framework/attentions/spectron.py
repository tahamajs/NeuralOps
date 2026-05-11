#!/usr/bin/env python3
"""
SpectronAttention – Importance‑Aware Low‑Rank + Sparse Attention
================================================================
Based on D‑Rank, ARSVD, SVD‑LLM V2, and LoSparse/3BASiL principles.
Dynamically allocates rank per weight matrix under a global parameter budget,
and optionally augments the low‑rank approximation with a trainable sparse residual.

Usage:
    python SpectronAttention.py --task sst2 --epochs 5 --batch_size 16 --lr 1e-5 --budget_ratio 0.2
    python SpectronAttention.py --task sst2 --epochs 5 --batch_size 16 --lr 1e-5 --budget_ratio 0.2 --add_sparse
"""

import os, argparse, logging, warnings, copy
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import svd

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback,
    set_seed,
)
from datasets import Dataset, DatasetDict

# --------------------- environment ---------------------------------
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

# ======================================================================
# 1. Spectral utilities
# ======================================================================

def spectral_entropy(singular_values: np.ndarray) -> float:
    """Normalised entropy of *squared* singular values."""
    s2 = singular_values.astype(np.float64) ** 2
    total = s2.sum()
    if total == 0:
        return 0.0
    prob = s2 / total
    prob = prob[prob > 1e-12]
    return float(-np.sum(prob * np.log(prob)))

def cumulative_entropy_fraction(s: np.ndarray, max_rank: int) -> np.ndarray:
    """Return array of length max_rank: fraction of total entropy retained by top k values."""
    s2 = s.astype(np.float64) ** 2
    total = s2.sum()
    if total == 0:
        return np.ones(max_rank)
    cum = np.cumsum(s2[:max_rank]) / total
    return cum

# ======================================================================
# 2. Lagrangian rank allocator (importance‑aware)
# ======================================================================

def allocate_ranks_spectron(
    weight_matrices: List[np.ndarray],
    target_budget: int,
    names: List[str],
    max_rank_factor: float = 0.75,
    lambda_range: Tuple[float, float] = (1e-8, 5.0),
    steps: int = 100,
) -> Dict[str, int]:
    """
    Minimise Σ cost_i(r_i) subject to Σ params_i(r_i) ≤ budget.
    cost_i = 1 - cum_entropy_fraction  (information loss)
    params_i = r * (d_in + d_out)
    """
    cost_tables, param_tables = [], []
    for W in weight_matrices:
        d_out, d_in = W.shape
        max_r = min(d_in, d_out, int(max_rank_factor * min(d_in, d_out)))
        max_r = max(1, max_r)
        _, s, _ = svd(W.astype(np.float64), full_matrices=False)
        cum_ent = cumulative_entropy_fraction(s, max_r)
        cost = 1.0 - cum_ent
        cost_tables.append(cost)
        param_tables.append((np.arange(1, max_r + 1) * (d_in + d_out)).astype(np.float64))

    low, high = lambda_range
    best_ranks = None
    for _ in range(steps):
        lam = (low + high) / 2.0
        ranks = []
        total_params = 0
        for i in range(len(weight_matrices)):
            obj = cost_tables[i] + lam * param_tables[i]
            idx = int(np.argmin(obj))
            ranks.append(idx + 1)
            total_params += param_tables[i][idx]
        if total_params <= target_budget:
            best_ranks = ranks.copy()
            high = lam
        else:
            low = lam
    if best_ranks is None:
        best_ranks = [1] * len(weight_matrices)
        log.warning("Budget too tight, fallback to rank=1 everywhere.")
    return {n: r for n, r in zip(names, best_ranks)}

# ======================================================================
# 3. Low‑Rank + Sparse linear layer
# ======================================================================

class LowRankSparseLinear(nn.Module):
    """
    Composite linear layer: output = (up @ down + sparse_weight) @ x + bias.
    sparse_weight is a dense parameter that is periodically pruned to stay sparse.
    """
    def __init__(self, down: nn.Linear, up: nn.Linear,
                 sparse_weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.down = down          # nn.Linear(in, rank, bias=False)
        self.up = up              # nn.Linear(rank, out, bias=True)
        self.sparse_weight = nn.Parameter(sparse_weight)
        self.bias = bias
        # store original up bias (already inside up)
        self.out_features, self.in_features = up.out_features, down.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # low‑rank path
        x_lr = self.up(self.down(x))                    # (B, N, out)
        # sparse path
        x_sp = F.linear(x, self.sparse_weight)          # sparse_weight is (out, in)
        return x_lr + x_sp

    def prune_sparse(self, sparsity: float):
        """Zero out smallest magnitude entries until desired sparsity is reached."""
        with torch.no_grad():
            w = self.sparse_weight.data.abs()
            if w.numel() == 0:
                return
            k = int(w.numel() * (1 - sparsity))        # keep top-k
            if k >= w.numel():
                return
            threshold = torch.kthvalue(w.view(-1), w.numel() - k).values
            mask = w >= threshold
            self.sparse_weight.data *= mask

# ======================================================================
# 4. SpectronAttention module
# ======================================================================

class SpectronAttention(nn.Module):
    """
    Importance‑aware SVD‑compressed attention with optional sparse residual.
    Replaces DistilBERT's multi‑head attention.
    """
    def __init__(
        self,
        model_config,
        dropout: float = 0.1,
        layer_idx: int = 0,
        pretrained_attn=None,
        ranks: Dict[str, int] = None,
        add_sparse: bool = False,
        sparsity_level: float = 0.7,
    ):
        super().__init__()
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        if pretrained_attn is None or ranks is None:
            raise ValueError("Both pretrained_attn and ranks dict required.")

        # factorise all projections with given ranks
        proj_weights = {
            'q': (pretrained_attn.q_lin.weight.data, pretrained_attn.q_lin.bias),
            'k': (pretrained_attn.k_lin.weight.data, pretrained_attn.k_lin.bias),
            'v': (pretrained_attn.v_lin.weight.data, pretrained_attn.v_lin.bias),
            'o': (pretrained_attn.out_lin.weight.data, pretrained_attn.out_lin.bias),
        }
        for proj in ['q','k','v','o']:
            W, bias = proj_weights[proj]
            down, up = self._factorise(W, bias, ranks[proj])
            setattr(self, f"{proj}_down", down)
            setattr(self, f"{proj}_up", up)
            if add_sparse:
                sparse = self._create_sparse_residual(W, down, up, sparsity_level)
                self.register_parameter(f"{proj}_sparse", sparse)

        self.add_sparse = add_sparse
        self.dropout = nn.Dropout(dropout)

    def _factorise(self, W: torch.Tensor, bias: torch.Tensor, rank: int):
        d_out, d_in = W.shape
        rank = min(rank, min(d_out, d_in))
        rank = max(rank, 1)
        U, s, VT = svd(W.cpu().numpy().astype(np.float64), full_matrices=False)
        U_r = U[:, :rank]
        S_r = np.diag(s[:rank])
        VT_r = VT[:rank, :]

        down_weight = torch.from_numpy(VT_r).float()            # (rank, d_in)
        up_weight = torch.from_numpy(U_r @ S_r).float()         # (d_out, rank)

        down = nn.Linear(d_in, rank, bias=False)
        up = nn.Linear(rank, d_out, bias=True)
        down.weight.data = down_weight
        up.weight.data = up_weight
        up.bias.data = bias.float() if bias is not None else torch.zeros(d_out)
        return down, up

    def _create_sparse_residual(self, W: torch.Tensor, down: nn.Linear,
                                up: nn.Linear, sparsity: float) -> nn.Parameter:
        with torch.no_grad():
            low_rank = up.weight @ down.weight     # (d_out, d_in)
            residual = W - low_rank
            residual_abs = residual.abs()
            k = int(residual.numel() * (1 - sparsity))
            if k > 0 and k < residual.numel():
                threshold = torch.kthvalue(residual_abs.view(-1), residual.numel() - k).values
                residual[residual_abs < threshold] = 0.0
        return nn.Parameter(residual)

    def _linear_proj(self, x, proj: str):
        down = getattr(self, f"{proj}_down")
        up = getattr(self, f"{proj}_up")
        out = up(down(x))
        if self.add_sparse:
            sparse = getattr(self, f"{proj}_sparse")
            out = out + F.linear(x, sparse)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        Q = self._linear_proj(hidden_states, 'q')
        K = self._linear_proj(hidden_states, 'k')
        V = self._linear_proj(hidden_states, 'v')

        # split heads
        Q = Q.view(B, N, H, d).transpose(1, 2).contiguous()
        K = K.view(B, N, H, d).transpose(1, 2).contiguous()
        V = V.view(B, N, H, d).transpose(1, 2).contiguous()

        scale = d ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            # اصلاح این بخش:
            # اگر ماسک ۲ بعدی است (B, N)، آن را به (B, 1, 1, N) تبدیل کن
            # اگر ۴ بعدی است، مستقیم جمع کن
            if attention_mask.dim() == 2:
                scores = scores + attention_mask[:, None, None, :]
            else:
                scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        # ... ادامه کد ...

        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        bs, hh, sl, hd = context.shape
        context = context.transpose(1, 2).contiguous().reshape(bs, sl, hh * hd)
        assert context.shape[-1] == self.all_head_size

        output = self._linear_proj(context, 'o')
        if output_attentions:
            return (output, attn_weights)
        return (output, None)

# ======================================================================
# 5. Injector & weight gathering
# ======================================================================

def gather_attention_weights(model) -> Tuple[List[np.ndarray], List[str]]:
    """Return flattened list of all Q/K/V/Out weight matrices and their names."""
    weights = []
    names = []
    for lid in range(6):
        attn = model.distilbert.transformer.layer[lid].attention
        for proj, attr in [('q','q_lin'), ('k','k_lin'), ('v','v_lin'), ('out','out_lin')]:
            w = getattr(attn, attr).weight.data.cpu().numpy()
            weights.append(w)
            names.append(f"layer{lid}_{proj}")
    return weights, names

def inject_spectron(model, rank_dict: Dict[str, int], args, config):
    for lid in range(6):
        distil_layer = model.distilbert.transformer.layer[lid]
        old_attn = distil_layer.attention
        ranks = {}
        for p, key in [('q','q'), ('k','k'), ('v','v'), ('o','out')]:
            ranks[p] = rank_dict[f"layer{lid}_{key}"]
        new_attn = SpectronAttention(
            model_config=config,
            dropout=args.dropout,
            layer_idx=lid,
            pretrained_attn=old_attn,
            ranks=ranks,
            add_sparse=args.add_sparse,
            sparsity_level=args.sparsity,
        )
        distil_layer.attention = new_attn

# ======================================================================
# 6. Sparse pruning callback (applied during training)
# ======================================================================

class SparsePruningCallback(TrainerCallback):
    """Periodically prune sparse weights to maintain target sparsity."""
    def __init__(self, model, prune_freq: int = 200, final_sparsity: float = 0.7):
        self.model = model
        self.prune_freq = prune_freq
        self.final_sparsity = final_sparsity

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.prune_freq == 0:
            for layer in self.model.distilbert.transformer.layer:
                attn = layer.attention
                if isinstance(attn, SpectronAttention) and attn.add_sparse:
                    for proj in ['q','k','v','o']:
                        sparse = getattr(attn, f"{proj}_sparse", None)
                        if sparse is not None:
                            with torch.no_grad():
                                w = sparse.data
                                k = int(w.numel() * (1 - self.final_sparsity))
                                if k >= w.numel(): continue
                                threshold = torch.kthvalue(w.abs().view(-1), w.numel() - k).values
                                w[w.abs() < threshold] = 0.0

# ======================================================================
# 7. GLUE data (unchanged)
# ======================================================================

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

def load_glue_local(task):
    from framework.utils.paths import get_glue_dir
    base = str(get_glue_dir())
    path = os.path.join(base, task)
    splits = {}
    for f in os.listdir(path):
        if f.endswith(".parquet"):
            s = f.split("-")[0]
            splits[s] = Dataset.from_parquet(os.path.join(path, f))
    return DatasetDict(splits)

def get_num_labels_and_type(task):
    if task == "stsb": return 1, True
    if task == "cola": return 2, False
    m = {"sst2":2,"mrpc":2,"qqp":2,"mnli":3,"qnli":2,"rte":2,"wnli":2}
    return m[task], False

def preprocess(examples, tokenizer, s1, s2, max_len, is_reg):
    if s2 is None:
        t = tokenizer(examples[s1], truncation=True, padding="max_length", max_length=max_len)
    else:
        t = tokenizer(examples[s1], examples[s2], truncation=True, padding="max_length", max_length=max_len)
    t["labels"] = [float(l) for l in examples["label"]] if is_reg else examples["label"]
    return t

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": (logits.argmax(-1) == labels).mean().item()}

# ======================================================================
# 8. Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="./spectron_output")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget_ratio", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--add_sparse", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--prune_freq", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    raw = load_glue_local(args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    num_labels, is_reg = get_num_labels_and_type(args.task)
    s1, s2 = GLUE_TASK_KEYS[args.task]
    tokenized = raw.map(lambda x: preprocess(x, tokenizer, s1, s2, args.max_len, is_reg),
                        batched=True, remove_columns=raw["train"].column_names)
    train_ds = tokenized["train"]
    eval_ds = tokenized.get("validation", tokenized.get("validation_matched"))
    log.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Model + rank allocation
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config, local_files_only=True)

    all_weights, all_names = gather_attention_weights(model)
    original_total = sum(w.size for w in all_weights)
    target = int(original_total * args.budget_ratio)
    log.info(f"Original attention params: {original_total:,} → target {args.budget_ratio*100:.0f}% = {target:,}")

    rank_dict = allocate_ranks_spectron(all_weights, target, all_names)
    avg_rank = np.mean(list(rank_dict.values()))
    log.info(f"Allocated ranks (avg {avg_rank:.1f}): {rank_dict}")

    inject_spectron(model, rank_dict, args, config)

    # Training args
    total_steps = (len(train_ds) // args.batch_size) * args.epochs
    warmup = int(0.06 * total_steps)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=warmup,
        logging_steps=100,
        eval_strategy="steps",
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

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    if args.add_sparse:
        callbacks.append(SparsePruningCallback(model, prune_freq=args.prune_freq, final_sparsity=args.sparsity))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    log.info("Starting training with SpectronAttention...")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, args.task))
    log.info("Training complete.")

if __name__ == "__main__":
    main()