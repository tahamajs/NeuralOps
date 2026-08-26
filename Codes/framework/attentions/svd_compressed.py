r"""
SVD-Compressed Attention with Importance-Aware Lagrangian Rank Allocation.

Truncated Singular Value Decomposition (SVD) attention compression module with
Lagrangian spectral-entropy optimal rank budgeting and optional low-rank plus sparse residual decomposition.

Mathematical Mechanism:
    For projection weight matrices $\{W_i\}_{i=1}^M$ across all transformer attention sub-layers:
    1. **Spectral Information Loss & Parameter Cost:**
       $$E_i(r) = \frac{\sum_{j=1}^r \sigma_{i, j}^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_{i, j}^2}, \quad \operatorname{Cost}_i(r) = 1 - E_i(r)$$
       $$\operatorname{Params}_i(r) = r \cdot (d_{i, \text{in}} + d_{i, \text{out}})$$
    2. **Lagrangian Global Rank Optimization:**
       $$\min_{\{r_i\}} \sum_{i=1}^M \operatorname{Cost}_i(r_i) \quad \text{subject to} \quad \sum_{i=1}^M \operatorname{Params}_i(r_i) \le B_{\text{target}}$$
    3. **Low-Rank Factorized Multi-Head Projections:**
       $$W_i \approx W_{i}^{\text{up}} W_{i}^{\text{down}}, \quad W_{i}^{\text{down}} \in \mathbb{R}^{r_i \times d_{\text{in}}}, \, W_{i}^{\text{up}} \in \mathbb{R}^{d_{\text{out}} \times r_i}$$
       $$\operatorname{Attn}(Q, K, V) = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) V$$

Computational Complexity:
    - Low-rank projection per token: $\mathcal{O}(d_{\text{in}} r_i + r_i d_{\text{out}})$.
    - Attention matrix evaluation: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot N \cdot (d_{\text{in}} r + r d_{\text{out}}) + B \cdot H \cdot N^2 \cdot d)$.
"""

import argparse
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from scipy.linalg import svd

try:
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except Exception:
    AutoConfig = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    EarlyStoppingCallback = object
    Trainer = object
    TrainingArguments = object
    def set_seed(s): pass

try:
    from datasets import Dataset, DatasetDict
except Exception:
    Dataset = object
    DatasetDict = object

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)


# =============================================================================
# 1. Helper functions for spectral analysis
# =============================================================================


def calculate_spectral_entropy(singular_values: np.ndarray) -> float:
    r"""Compute normalized spectral entropy of singular value distribution."""
    s = singular_values.astype(np.float64)
    s_sum = s.sum()
    if s_sum == 0:
        return 0.0
    prob = s / s_sum
    prob = prob[prob > 1e-5]
    return float(-np.sum(prob * np.log(prob)))


def compute_cumulative_entropy(
    singular_values: np.ndarray, max_rank: int
) -> np.ndarray:
    r"""Compute cumulative spectral energy fraction retained by top-$k$ singular values."""
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
    weight_matrices: List[np.ndarray],
    target_budget: int,
    layer_proj_names: List[str],
    max_rank_factor: float = 0.5,
    lambda_bracket: Tuple[float, float] = (1e-5, 10.0),
    num_lambda_steps: int = 100,
) -> Dict[str, int]:
    r"""Allocate per-projection ranks under global parameter constraint via binary search on Lagrange multiplier."""
    cost_table = []
    param_table = []

    for W in weight_matrices:
        d_in, d_out = W.shape[1], W.shape[0]
        max_r = min(d_in, d_out, int(max_rank_factor * min(d_in, d_out)))
        max_r = max(1, max_r)

        _, s, _ = svd(W.astype(np.float64), full_matrices=False)
        cum_ent = compute_cumulative_entropy(s, max_r)
        cost = 1.0 - cum_ent
        cost_table.append(cost)

        params_per_r = np.arange(1, max_r + 1) * (d_in + d_out)
        param_table.append(params_per_r)

    low, high = lambda_bracket
    best_allocation = None

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
            high = lam
        else:
            low = lam

    if best_allocation is None:
        best_allocation = [1] * len(weight_matrices)
        log.warning("Could not meet budget, falling back to rank=1 everywhere.")

    rank_dict = {name: r for name, r in zip(layer_proj_names, best_allocation)}
    return rank_dict


# =============================================================================
# 3. SVD‑Compressed Attention module
# =============================================================================


@ATTENTION_REGISTRY.register("svd_compressed_module")
@ATTENTION_REGISTRY.register("svd_compressed")
class SVDCompressedAttention(BaseAttention):
    r"""Multi-head attention with truncated SVD projections and optional sparse residuals.

    Args:
        model_config: Transformer model configuration.
        dropout: Dropout rate for attention weights.
        layer_idx: Layer index in transformer network.
        pretrained_attn: Source attention module for weight factorizations.
        ranks: Dictionary containing target ranks for `q`, `k`, `v`, `o`.
        add_sparse: Whether to instantiate sparse residual parameters.
        sparsity_level: Target sparsity fraction $\rho \in (0, 1)$.
    """

    def __init__(
        self,
        model_config: Any,
        dropout: float = 0.1,
        layer_idx: int = 0,
        pretrained_attn: Optional[Any] = None,
        ranks: Optional[Dict[str, int]] = None,
        add_sparse: bool = False,
        sparsity_level: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config)
        self.num_heads: int = getattr(model_config, "num_attention_heads", 12)
        self.head_dim: int = getattr(model_config, "hidden_size", 768) // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = getattr(model_config, "hidden_size", 768)

        default_r = getattr(model_config, "svd_rank", getattr(model_config, "rank", 64))
        self.ranks: Dict[str, int] = ranks or {"q": default_r, "k": default_r, "v": default_r, "o": default_r}
        self.add_sparse: bool = add_sparse
        self.sparsity_level: float = sparsity_level

        self.q_down = nn.Linear(self.hidden_size, self.ranks["q"], bias=False)
        self.q_up = nn.Linear(self.ranks["q"], self.all_head_size, bias=True)
        self.k_down = nn.Linear(self.hidden_size, self.ranks["k"], bias=False)
        self.k_up = nn.Linear(self.ranks["k"], self.all_head_size, bias=True)
        self.v_down = nn.Linear(self.hidden_size, self.ranks["v"], bias=False)
        self.v_up = nn.Linear(self.ranks["v"], self.all_head_size, bias=True)
        self.o_down = nn.Linear(self.all_head_size, self.ranks["o"], bias=False)
        self.o_up = nn.Linear(self.ranks["o"], self.hidden_size, bias=True)

        for mod in [self.q_down, self.q_up, self.k_down, self.k_up, self.v_down, self.v_up, self.o_down, self.o_up]:
            if isinstance(mod, nn.Linear):
                nn.init.orthogonal_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

        if add_sparse:
            self.q_sparse_weight = nn.Parameter(torch.zeros(self.all_head_size, self.hidden_size))
            self.k_sparse_weight = nn.Parameter(torch.zeros(self.all_head_size, self.hidden_size))
            self.v_sparse_weight = nn.Parameter(torch.zeros(self.all_head_size, self.hidden_size))
            self.o_sparse_weight = nn.Parameter(torch.zeros(self.hidden_size, self.all_head_size))
        else:
            self.register_parameter("q_sparse_weight", None)
            self.register_parameter("k_sparse_weight", None)
            self.register_parameter("v_sparse_weight", None)
            self.register_parameter("o_sparse_weight", None)

        dropout_p = getattr(
            model_config,
            "dropout",
            getattr(
                model_config,
                "hidden_dropout_prob",
                getattr(model_config, "attention_probs_dropout_prob", dropout),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _set_factorised_weights(
        self, W_tensor: torch.Tensor, bias_tensor: Optional[torch.Tensor], down: nn.Linear, up: nn.Linear, rank: int
    ) -> None:
        with torch.no_grad():
            W = W_tensor.float()
            d_out, d_in = W.shape
            r = min(rank, min(d_out, d_in), down.out_features, up.in_features)
            r = max(r, 1)

            U, s, VT = torch.linalg.svd(W, full_matrices=False)
            sqrt_s = torch.sqrt(torch.clamp(s[:r], min=1e-12))
            down_weight = (sqrt_s.unsqueeze(1) * VT[:r, :]).to(device=down.weight.device, dtype=down.weight.dtype)
            up_weight = (U[:, :r] * sqrt_s.unsqueeze(0)).to(device=up.weight.device, dtype=up.weight.dtype)

            down.weight.data.zero_()
            up.weight.data.zero_()
            down.weight.data[:r, :].copy_(down_weight)
            up.weight.data[:, :r].copy_(up_weight)
            if bias_tensor is not None and up.bias is not None:
                up.bias.data.copy_(bias_tensor.to(device=up.bias.device, dtype=up.bias.dtype))

    def copy_weights(self, src: Any) -> None:
        """Decompose and copy weights from any HuggingFace or PyTorch MHA module."""
        from framework.core.adapters import UniversalAdapter

        mapping = [
            ("q", self.q_down, self.q_up, self.ranks["q"]),
            ("k", self.k_down, self.k_up, self.ranks["k"]),
            ("v", self.v_down, self.v_up, self.ranks["v"]),
            ("o", self.o_down, self.o_up, self.ranks["o"]),
        ]
        for proj, down, up, r in mapping:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                self._set_factorised_weights(w, b, down, up, r)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

    def _create_pruned_residual(
        self, W: np.ndarray, down: nn.Linear, up: nn.Linear, sparsity: float
    ) -> nn.Parameter:
        r"""Construct dense parameter holding pruned residual values."""
        low_rank = (up.weight.data @ down.weight.data).cpu().numpy()
        residual = W - low_rank
        threshold = np.percentile(np.abs(residual), sparsity * 100)
        sparse_res = np.where(np.abs(residual) >= threshold, residual, 0.0)
        return nn.Parameter(torch.from_numpy(sparse_res).float())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Forward pass computing SVD-compressed multi-head attention.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head pruning tensor.
            output_attentions: Return explicit attention probabilities if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax probability matrix or None.
        """
        B, N, _ = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        def proj_q(x: torch.Tensor) -> torch.Tensor:
            """Proj q."""
            out = self.q_up(self.q_down(x))
            if self.add_sparse:
                out = out + F.linear(x, self.q_sparse_weight)
            return out

        def proj_k(x: torch.Tensor) -> torch.Tensor:
            """Proj k."""
            out = self.k_up(self.k_down(x))
            if self.add_sparse:
                out = out + F.linear(x, self.k_sparse_weight)
            return out

        def proj_v(x: torch.Tensor) -> torch.Tensor:
            """Proj v."""
            out = self.v_up(self.v_down(x))
            if self.add_sparse:
                out = out + F.linear(x, self.v_sparse_weight)
            return out

        def proj_o(x: torch.Tensor) -> torch.Tensor:
            """Proj o."""
            out = self.o_up(self.o_down(x))
            if self.add_sparse:
                out = out + F.linear(x, self.o_sparse_weight)
            return out

        Q = proj_q(hidden_states)
        K = proj_k(hidden_states)
        V = proj_v(hidden_states)

        Q = Q.view(B, N, H, d).transpose(1, 2).contiguous()
        K = K.view(B, N, H, d).transpose(1, 2).contiguous()
        V = V.view(B, N, H, d).transpose(1, 2).contiguous()

        scale = d**-0.5
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask) * -10000.0
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1)
            else:
                scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        output = proj_o(context)

        if output_attentions:
            return output, attn_weights
        return output, None


# =============================================================================
# 4. Injection helpers
# =============================================================================


def gather_all_attention_weights(model: Any) -> Tuple[List[np.ndarray], List[str]]:
    r"""Gather attention projection weights and layer names across model layers."""
    weights = []
    names = []
    for layer_idx in range(len(model.distilbert.transformer.layer)):
        distil_layer = model.distilbert.transformer.layer[layer_idx]
        attn = distil_layer.attention
        for proj in ["q_lin", "k_lin", "v_lin", "out_lin"]:
            w = getattr(attn, proj).weight.data.cpu().numpy()
            weights.append(w)
            proj_short = proj.replace("_lin", "")
            names.append(f"layer{layer_idx}_{proj_short}")
    return weights, names


def inject_all_layers(
    model: Any,
    rank_dict: Dict[str, int],
    dropout: float,
    add_sparse: bool,
    sparsity: float,
    config: Any,
    **kwargs) -> None:
    r"""Inject SVDCompressedAttention into all transformer layers of model."""
    for layer_idx in range(len(model.distilbert.transformer.layer)):
        distil_layer = model.distilbert.transformer.layer[layer_idx]
        old_attn = distil_layer.attention
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
# 5. GLUE data loading
# =============================================================================

GLUE_TASK_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
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
    r"""Load GLUE benchmark dataset from local offline storage."""
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


def get_num_labels_and_is_regression(task: str) -> Tuple[int, bool]:
    r"""Return class count and regression indicator for GLUE task."""
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
    examples: Any,
    tokenizer: Any,
    sent1_key: str,
    sent2_key: Optional[str],
    max_length: int,
    is_regression: bool,
) -> Dict[str, Any]:
    r"""Preprocess and tokenize dataset sequence examples."""
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


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    r"""Compute classification evaluation accuracy."""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": float((preds == labels).mean().item())}


# =============================================================================
# 6. Main training script
# =============================================================================


def main() -> None:
    r"""Run full training workflow for SVDCompressedAttention on GLUE."""
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
        help="Target fraction of original attention parameters (e.g. 0.2 = 20%)",
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

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=config, local_files_only=True
    )

    all_weights, all_names = gather_all_attention_weights(model)
    original_total_params = sum(w.size for w in all_weights)
    target_budget = int(original_total_params * args.budget_ratio)
    log.info(
        f"Original attention parameters: {original_total_params:,}, "
        f"target budget ({args.budget_ratio*100:.0f}%): {target_budget:,}"
    )

    rank_dict = lagrangian_rank_allocation(
        weight_matrices=all_weights,
        target_budget=target_budget,
        layer_proj_names=all_names,
    )
    avg_rank = float(np.mean(list(rank_dict.values())))
    log.info(f"Optimised ranks across layers: average rank = {avg_rank:.1f}")

    inject_all_layers(
        model,
        rank_dict,
        args.dropout,
        add_sparse=args.add_sparse,
        sparsity=args.sparsity,
        config=config,
    )
    log.info("Injected importance‑aware SVD‑Compressed Attention.")

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


@ATTENTION_REGISTRY.register("svd_compressed_inject")
def inject_svd_compressed(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
    **kwargs) -> SVDCompressedAttention:
    r"""Bridge helper injecting SVDCompressedAttention into a transformer layer."""
    rank = getattr(comp_config, "rank", 32)
    dropout = getattr(comp_config, "dropout", 0.1)
    ranks = {"q": rank, "k": rank, "v": rank, "o": rank}
    new_attn = SVDCompressedAttention(
        model_config,
        dropout=dropout,
        layer_idx=layer_idx,
        pretrained_attn=old_attn,
        ranks=ranks,
    )
    layer.attention = new_attn
    return new_attn
