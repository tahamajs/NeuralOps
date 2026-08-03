#!/usr/bin/env python3
"""
SVD‑Compressed Attention – Production Version with
Importance‑Aware Rank Allocation (D‑Rank / ARSVD style)
and optional Low‑Rank + Sparse Decomposition.

Usage:
    python SVDCompressedAttention.py --task sst2 --epochs 5 --batch_size 16 --lr 1e-5
"""

import argparse
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from scipy.linalg import svd
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from datasets import Dataset, DatasetDict

# ~~~~~~~~ Offline environment ~~~~~~~~
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

# =============================================================================
# 1. Helper functions for spectral analysis
# =============================================================================


def calculate_spectral_entropy(singular_values: np.ndarray) -> float:
    """Normalised spectral entropy (higher = more info spread)."""
    s = singular_values.astype(np.float64)
    s_sum = s.sum()
    if s_sum == 0:
        return 0.0
    prob = s / s_sum
    prob = prob[prob > 1e-12]  # ignore zeros
    return float(-np.sum(prob * np.log(prob)))


def compute_cumulative_entropy(
    singular_values: np.ndarray, max_rank: int
) -> np.ndarray:
    """
    Returns array `cum_ent[k]` = fraction of total entropy retained
    when using the first k singular values (k=1..max_rank).
    """
    s = singular_values.astype(np.float64)
    s_sq = s**2
    total = s_sq.sum()
    if total == 0:
        return np.ones(max_rank)
    cum = np.cumsum(s_sq[:max_rank]) / total
    return cum


# =============================================================================
# 2. Lagrangian optimisation for rank allocation
# =============================================================================


def lagrangian_rank_allocation(
    weight_matrices: List[
        np.ndarray
    ],  # list of all original weight matrices (Q,K,V,O × layers)
    target_budget: int,
    layer_proj_names: List[str],  # e.g. ['layer0_q','layer0_k',...]
    max_rank_factor: float = 0.5,  # max rank as fraction of min(d_in,d_out)
    lambda_bracket: Tuple[float, float] = (1e-8, 10.0),
    num_lambda_steps: int = 100,
) -> Dict[str, int]:
    """
    Solves: min Σ cost_i(r_i)  s.t. Σ params_i(r_i) ≤ budget
    using a Lagrangian multiplier λ.

    cost_i(r) = 1 - cum_entropy_fraction(r)   (lower = better)
    params_i(r) = r * (d_in + d_out)          (approximate parameter count)
    """
    # Pre‑compute for each matrix the cost and param count for every possible rank
    max_ranks = []
    cost_table = []  # list of arrays shape (max_rank,)
    param_table = []  # list of arrays shape (max_rank,)

    for W, name in zip(weight_matrices, layer_proj_names):
        d_in, d_out = W.shape[1], W.shape[0]
        max_r = min(d_in, d_out, int(max_rank_factor * min(d_in, d_out)))
        max_r = max(1, max_r)
        max_ranks.append(max_r)

        # singular values
        _, s, _ = svd(W.astype(np.float64), full_matrices=False)
        cum_ent = compute_cumulative_entropy(s, max_r)  # length max_r
        cost = 1.0 - cum_ent  # information loss
        cost_table.append(cost)

        # params per rank: both down (d_in→r) and up (r→d_out)
        params_per_r = np.arange(1, max_r + 1) * (d_in + d_out)
        param_table.append(params_per_r)

    # Binary search on λ
    low, high = lambda_bracket
    best_allocation = None

    for _ in range(num_lambda_steps):
        lam = (low + high) / 2.0

        # For each matrix, pick rank that minimises cost + λ * params
        ranks = []
        total_params = 0
        for i in range(len(weight_matrices)):
            obj = cost_table[i] + lam * param_table[i]
            best_r_idx = int(np.argmin(obj))
            r = best_r_idx + 1  # rank = index+1
            ranks.append(r)
            total_params += param_table[i][best_r_idx]

        if total_params <= target_budget:
            # feasible – try smaller λ (less penalty on params → maybe more params)
            best_allocation = ranks.copy()
            high = lam
        else:
            low = lam

    if best_allocation is None:
        # fallback: smallest rank everywhere
        best_allocation = [1] * len(weight_matrices)
        log.warning("Could not meet budget, falling back to rank=1 everywhere.")

    # Map back to dict
    rank_dict = {name: r for name, r in zip(layer_proj_names, best_allocation)}
    return rank_dict


# =============================================================================
# 3. SVD‑Compressed Attention module (updated)
# =============================================================================


@ATTENTION_REGISTRY.register("svd_compressed")
class SVDCompressedAttention(BaseAttention):
    """
    Replaces a DistilBERT multi‑head attention layer with SVD‑factorised
    Q/K/V/O projections. Ranks are provided externally (pre‑computed).
    """

    def __init__(
        self,
        model_config,
        dropout: float = 0.1,
        layer_idx: int = 0,
        pretrained_attn=None,
        ranks: Optional[Dict[str, int]] = None,  # {'q':rq, 'k':rk, 'v':rv, 'o':ro}
        add_sparse: bool = False,  # future: low‑rank + sparse
        sparsity_level: float = 0.7,
    ):
        super().__init__(model_config)
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size

        if pretrained_attn is None:
            raise ValueError("pretrained_attn is required for weight init")
        if ranks is None:
            raise ValueError(
                "ranks dictionary is required (containing 'q','k','v','o')"
            )

        # ---- Factorise Q, K, V, O with given ranks ----
        q_w = pretrained_attn.q_lin.weight.data.cpu().numpy()
        k_w = pretrained_attn.k_lin.weight.data.cpu().numpy()
        v_w = pretrained_attn.v_lin.weight.data.cpu().numpy()
        out_w = pretrained_attn.out_lin.weight.data.cpu().numpy()

        self.q_down, self.q_up = self._factorise_linear(
            q_w, pretrained_attn.q_lin.bias, ranks["q"]
        )
        self.k_down, self.k_up = self._factorise_linear(
            k_w, pretrained_attn.k_lin.bias, ranks["k"]
        )
        self.v_down, self.v_up = self._factorise_linear(
            v_w, pretrained_attn.v_lin.bias, ranks["v"]
        )
        self.o_down, self.o_up = self._factorise_linear(
            out_w, pretrained_attn.out_lin.bias, ranks["o"]
        )

        # Sparse additive component (optional)
        self.add_sparse = add_sparse
        if add_sparse:
            self.q_sparse_weight = self._create_pruned_residual(
                q_w, self.q_down, self.q_up, sparsity_level
            )
            self.k_sparse_weight = self._create_pruned_residual(
                k_w, self.k_down, self.k_up, sparsity_level
            )
            self.v_sparse_weight = self._create_pruned_residual(
                v_w, self.v_down, self.v_up, sparsity_level
            )
            self.o_sparse_weight = self._create_pruned_residual(
                out_w, self.o_down, self.o_up, sparsity_level
            )

        self.dropout = nn.Dropout(dropout)

    def _factorise_linear(self, W: np.ndarray, bias: torch.Tensor, rank: int):
        """SVD factorisation W ≈ U_r Σ_r V_r^T → down (in→r), up (r→out)"""
        d_out, d_in = W.shape
        rank = min(rank, min(d_out, d_in))
        rank = max(rank, 1)

        U, s, VT = svd(W.astype(np.float64), full_matrices=False)
        U_r = U[:, :rank]
        S_r = np.diag(s[:rank])
        VT_r = VT[:rank, :]

        # down: in_features → rank  (weight shape [rank, in_features])
        down_weight = torch.from_numpy(VT_r).float()
        # up:   rank → out_features (weight shape [out_features, rank])
        up_weight = torch.from_numpy(U_r @ S_r).float()

        down = nn.Linear(d_in, rank, bias=False)
        up = nn.Linear(rank, d_out, bias=True)

        down.weight.data = down_weight
        up.weight.data = up_weight
        if bias is not None:
            up.bias.data = bias.data.float()
        else:
            up.bias.data.zero_()
        return down, up

    def _create_pruned_residual(
        self, W: np.ndarray, down: nn.Linear, up: nn.Linear, sparsity: float
    ) -> nn.Parameter:
        """Create a dense parameter containing the sparse residual."""
        low_rank = (up.weight.data @ down.weight.data).cpu().numpy()  # (out, in)
        residual = W - low_rank
        threshold = np.percentile(np.abs(residual), sparsity * 100)
        sparse_res = np.where(np.abs(residual) >= threshold, residual, 0.0)
        return nn.Parameter(torch.from_numpy(sparse_res.T).float())  # [out, in]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        def proj_q(x):
            out = self.q_up(self.q_down(x))
            if self.add_sparse:
                out = out + F.linear(x, self.q_sparse_weight)
            return out

        def proj_k(x):
            out = self.k_up(self.k_down(x))
            if self.add_sparse:
                out = out + F.linear(x, self.k_sparse_weight)
            return out

        def proj_v(x):
            out = self.v_up(self.v_down(x))
            if self.add_sparse:
                out = out + F.linear(x, self.v_sparse_weight)
            return out

        def proj_o(x):
            out = self.o_up(self.o_down(x))
            if self.add_sparse:
                out = out + F.linear(x, self.o_sparse_weight)
            return out

        Q = proj_q(hidden_states)  # (B, N, 768)
        K = proj_k(hidden_states)
        V = proj_v(hidden_states)

        Q = Q.view(B, N, H, d).transpose(1, 2).contiguous()  # (B, H, N, d)
        K = K.view(B, N, H, d).transpose(1, 2).contiguous()
        V = V.view(B, N, H, d).transpose(1, 2).contiguous()

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (d**0.5)  # (B, H, N, N)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)  # (B, H, N, d)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        output = proj_o(context)

        if output_attentions:
            return output, attn_weights
        return output, None


# =============================================================================
# 4. Injection helpers
# =============================================================================


def gather_all_attention_weights(model) -> Tuple[list, list]:
    """
    Returns:
        weights: list of np.ndarray (Q/K/V/O weight for each layer, in order)
        names:   list of strings like 'layer0_q','layer0_k',...
    """
    weights = []
    names = []
    for layer_idx in range(6):
        distil_layer = model.distilbert.transformer.layer[layer_idx]
        attn = distil_layer.attention
        for proj in ["q_lin", "k_lin", "v_lin", "out_lin"]:
            w = getattr(attn, proj).weight.data.cpu().numpy()
            weights.append(w)
            # short name
            proj_short = proj.replace("_lin", "")  # q, k, v, out
            names.append(f"layer{layer_idx}_{proj_short}")
    return weights, names


def inject_all_layers(
    model,
    rank_dict: Dict[str, int],
    dropout: float,
    add_sparse: bool,
    sparsity: float,
    config,
):
    """
    Replace every layer's attention with SVDCompressedAttention using
    the pre‑computed ranks.
    """
    for layer_idx in range(6):
        distil_layer = model.distilbert.transformer.layer[layer_idx]
        old_attn = distil_layer.attention
        # Extract ranks for this layer, mapping 'out' -> 'o' as needed
        ranks = {}
        for proj in ["q", "k", "v", "out"]:
            key = f"layer{layer_idx}_{proj}"
            if proj == "out":
                ranks["o"] = rank_dict[key]
            else:
                ranks[proj] = rank_dict[key]
        new_attn = SVDCompressedAttention(
            model_config=config,
            dropout=dropout,
            layer_idx=layer_idx,
            pretrained_attn=old_attn,
            ranks=ranks,
            add_sparse=add_sparse,
            sparsity_level=sparsity,
        )
        distil_layer.attention = new_attn


# =============================================================================
# 5. GLUE data loading (unchanged)
# =============================================================================

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
    mapping = {
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
    }
    return mapping[task], False


def preprocess_function(
    examples, tokenizer, sent1_key, sent2_key, max_length, is_regression
):
    if sent2_key is None:
        t = tokenizer(
            examples[sent1_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    else:
        t = tokenizer(
            examples[sent1_key],
            examples[sent2_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    t["labels"] = (
        [float(l) for l in examples["label"]] if is_regression else examples["label"]
    )
    return t


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": (preds == labels).mean().item()}


# =============================================================================
# 6. Main training script
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train DistilBERT with Importance‑Aware SVD‑Compressed Attention"
    )
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./svd_compressed_output")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budget_ratio",
        type=float,
        default=0.2,
        help="Target fraction of original attention parameters (e.g. 0.2 = 20%%)",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--add_sparse",
        action="store_true",
        help="Enable low‑rank + sparse decomposition (experimental)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.7,
        help="Sparsity level for residual (if --add_sparse)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    raw_datasets = load_glue_local(args.task)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True
    )
    num_labels, is_regression = get_num_labels_and_is_regression(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    tokenized = raw_datasets.map(
        lambda ex: preprocess_function(
            ex, tokenizer, sent1_key, sent2_key, args.max_len, is_regression
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    train_dataset = tokenized["train"]
    eval_dataset = tokenized.get("validation", tokenized.get("validation_matched"))
    log.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # ---- Model & rank optimisation ----
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=config, local_files_only=True
    )

    # 1. Gather all attention weights
    all_weights, all_names = gather_all_attention_weights(model)
    original_total_params = sum(w.size for w in all_weights)
    target_budget = int(original_total_params * args.budget_ratio)
    log.info(
        f"Original attention parameters: {original_total_params:,}, "
        f"target budget ({args.budget_ratio*100:.0f}%): {target_budget:,}"
    )

    # 2. Global rank optimisation
    rank_dict = lagrangian_rank_allocation(
        weight_matrices=all_weights,
        target_budget=target_budget,
        layer_proj_names=all_names,
    )
    avg_rank = np.mean(list(rank_dict.values()))
    log.info(f"Optimised ranks across layers: average rank = {avg_rank:.1f}")

    # 3. Inject compressed attention into all layers
    inject_all_layers(
        model,
        rank_dict,
        args.dropout,
        add_sparse=args.add_sparse,
        sparsity=args.sparsity,
        config=config,
    )
    log.info("Injected importance‑aware SVD‑Compressed Attention.")

    # ---- Training ----
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info("Starting training with importance‑aware SVD‑Compressed Attention...")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, args.task))
    log.info("Training completed.")


@ATTENTION_REGISTRY.register("svd_compressed")
def inject_svd_compressed(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    rank = getattr(comp_config, "rank", 32)
    ranks = {"q": rank, "k": rank, "v": rank, "o": rank}
    new_attn = SVDCompressedAttention(
        model_config,
        dropout=comp_config.dropout,
        layer_idx=layer_idx,
        pretrained_attn=old_attn,
        ranks=ranks,
    )
    layer.attention = new_attn
    return new_attn  # 20:12:35 | INFO    | Train size: 67349, Eval size: 872


# 20:12:36 | INFO    | HTTP Request: HEAD https://hf.devneeds.ir/distilbert-base-uncased/resolve/main/adapter_config.json "HTTP/1.1 404 Remote Manually Blocked"
# Loading weights: 100%|████████████████████████████████████████████████████| 100/100 [00:00<00:00, 4832.76it/s]
# [transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
# Key                     | Status     |
# ------------------------+------------+-
# vocab_layer_norm.weight | UNEXPECTED |
# vocab_layer_norm.bias   | UNEXPECTED |
# vocab_projector.bias    | UNEXPECTED |
# vocab_transform.bias    | UNEXPECTED |
# vocab_transform.weight  | UNEXPECTED |
# classifier.bias         | MISSING    |
# pre_classifier.bias     | MISSING    |
# classifier.weight       | MISSING    |
# pre_classifier.weight   | MISSING    |

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# 20:12:36 | INFO    | Original attention parameters: 14,155,776, target budget (20%): 2,831,155
# 20:12:39 | INFO    | Optimised ranks across layers: average rank = 76.8
# 20:12:41 | INFO    | Injected importance‑aware SVD‑Compressed Attention.
# 20:12:42 | INFO    | Starting training with importance‑aware SVD‑Compressed Attention...
# {'loss': '0.691', 'grad_norm': '1.017', 'learning_rate': '7.845e-07', 'epoch': '0.02375'}
# {'loss': '0.6867', 'grad_norm': '1.574', 'learning_rate': '1.577e-06', 'epoch': '0.04751'}
# {'loss': '0.6811', 'grad_norm': '2.553', 'learning_rate': '2.369e-06', 'epoch': '0.07126'}
# {'loss': '0.6705', 'grad_norm': '3.794', 'learning_rate': '3.162e-06', 'epoch': '0.09501'}
# {'loss': '0.6315', 'grad_norm': '30.41', 'learning_rate': '3.954e-06', 'epoch': '0.1188'}
# {'eval_loss': '0.7297', 'eval_accuracy': '0.5608', 'eval_runtime': '12.33', 'eval_samples_per_second': '70.69', 'eval_steps_per_second': '2.27', 'epoch': '0.1188'}
# Writing model shards: 100%|█████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.25s/it]
# {'loss': '0.6342', 'grad_norm': '42.53', 'learning_rate': '4.746e-06', 'epoch': '0.1425'}
# {'loss': '0.6048', 'grad_norm': '48.68', 'learning_rate': '5.539e-06', 'epoch': '0.1663'}
# {'loss': '0.5613', 'grad_norm': '66.09', 'learning_rate': '6.331e-06', 'epoch': '0.19'}
# {'loss': '0.5219', 'grad_norm': '24.1', 'learning_rate': '7.124e-06', 'epoch': '0.2138'}
# {'loss': '0.4681', 'grad_norm': '10.72', 'learning_rate': '7.916e-06', 'epoch': '0.2375'}
# {'eval_loss': '0.5233', 'eval_accuracy': '0.7626', 'eval_runtime': '10.24', 'eval_samples_per_second': '85.16', 'eval_steps_per_second': '2.735', 'epoch': '0.2375'}
# Writing model shards: 100%|█████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.34s/it]
# {'loss': '0.4643', 'grad_norm': '8.909', 'learning_rate': '8.708e-06', 'epoch': '0.2613'}
# {'loss': '0.3996', 'grad_norm': '20.39', 'learning_rate': '9.501e-06', 'epoch': '0.285'}
# {'loss': '0.4163', 'grad_norm': '23.58', 'learning_rate': '9.981e-06', 'epoch': '0.3088'}
# {'loss': '0.3874', 'grad_norm': '12.56', 'learning_rate': '9.931e-06', 'epoch': '0.3325'}
# {'loss': '0.3693', 'grad_norm': '7.309', 'learning_rate': '9.88e-06', 'epoch': '0.3563'}
# {'eval_loss': '0.4301', 'eval_accuracy': '0.805', 'eval_runtime': '9.497', 'eval_samples_per_second': '91.82', 'eval_steps_per_second': '2.948', 'epoch': '0.3563'}
# Writing model shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.01s/it]
# {'loss': '0.3538', 'grad_norm': '10.36', 'learning_rate': '9.83e-06', 'epoch': '0.38'}
# {'loss': '0.3574', 'grad_norm': '10.15', 'learning_rate': '9.779e-06', 'epoch': '0.4038'}
#   8%|███████████▎                                                                                                                          | 1771/21050 [18:02<2:52:47,  1.86it/s]^CTraceback (most recent call last):
#   File "/Users/tahamajs/Documents/uni/Project/Codes/SVDCompressedAttention.py", line 516, in <module>
#     main()
#     ~~~~^^
#   File "/Users/tahamajs/Documents/uni/Project/Codes/SVDCompressedAttention.py", line 511, in main
#     trainer.train()
#     ~~~~~~~~~~~~~^^
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 1427, in train
#     return inner_training_loop(
#         args=args,
#     ...<2 lines>...
#         ignore_keys_for_eval=ignore_keys_for_eval,
#     )
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 1509, in _inner_training_loop
#     self._run_epoch(
#     ~~~~~~~~~~~~~~~^
#         model=model,
#         ^^^^^^^^^^^^
#     ...<9 lines>...
#         steps_trained_in_current_epoch=steps_trained_in_current_epoch,
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#     )
#     ^
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 1742, in _run_epoch
#     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
#                                       ~~~~~~~~~~~^^^^^^^^^^^^^^
# KeyboardInterrupt
#   8%|███████████▎                                                                                                                          | 1771/21050 [18:04<3:16:50,  1.63it/s]
