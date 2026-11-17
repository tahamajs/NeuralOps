from .spectron.utils import (
    spectral_entropy,
    cumulative_entropy_fraction,
    allocate_ranks_spectron,
)
from .spectron.linear import LowRankSparseLinear
from .spectron.layer import (
    SpectronAttention,
    inject_spectron,
    SparsePruningCallback,
)

# Constants and helpers from original file
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

def get_num_labels_and_type(task: str) -> tuple[int, bool]:
    if task == "stsb":
        return 1, True
    if task == "cola":
        return 2, False
    return {"sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2}[task], False

def preprocess(examples, tokenizer, s1, s2, max_len, is_reg):
    if s2 is None:
        t = tokenizer(examples[s1], truncation=True, padding="max_length", max_length=max_len)
    else:
        t = tokenizer(
            examples[s1],
            examples[s2],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
    t["labels"] = [float(l) for l in examples["label"]] if is_reg else examples["label"]
    return t

def gather_attention_weights(model: Any) -> tuple[list[np.ndarray], list[str]]:
    import torch.nn as nn
    import numpy as np
    weights: list[np.ndarray] = []
    names: list[str] = []
    
    def _collect_attention_block(attn, layer_id):
        if hasattr(attn, "c_attn") and hasattr(attn, "c_proj"):
            w = attn.c_attn.weight.data.cpu().numpy()
            if w.shape[0] < w.shape[1]: w = w.T
            q_d = w.shape[0] // 3
            for tag, part in zip(("q", "k", "v"), (w[:q_d], w[q_d : 2 * q_d], w[2 * q_d :])):
                weights.append(part); names.append(f"layer{layer_id}_{tag}")
            weights.append(attn.c_proj.weight.data.cpu().numpy()); names.append(f"layer{layer_id}_o")
            return
        inner = getattr(attn, "self", attn)
        for proj, attr in (("q", "query"), ("k", "key"), ("v", "value")):
            if hasattr(inner, attr):
                w = getattr(inner, attr).weight.data.cpu().numpy()
                weights.append(w); names.append(f"layer{layer_id}_{proj}")
        for attr in ("dense", "out_proj", "o_proj", "out_lin"):
            if hasattr(attn, attr):
                weights.append(getattr(attn, attr).weight.data.cpu().numpy()); names.append(f"layer{layer_id}_o"); break
        for proj, names_ in (("q", ("q_proj", "q_lin")), ("k", ("k_proj", "k_lin")), ("v", ("v_proj", "v_lin"))):
            if not any(n.endswith(f"_{proj}") for n in names if f"_{proj}" in n):
                for nm in names_:
                    if hasattr(attn, nm):
                        weights.append(getattr(attn, nm).weight.data.cpu().numpy()); names.append(f"layer{layer_id}_{proj}"); break

    layer_lists = []
    for path in (("distilbert", "transformer", "layer"), ("roberta", "encoder", "layer"), ("bert", "encoder", "layer"), ("transformer", "h"), ("model", "layers"), ("encoder", "layer"), ("layers",), ("h",)):
        obj = model
        ok = True
        for p in path:
            if hasattr(obj, p): obj = getattr(obj, p)
            else: ok = False; break
        if ok and isinstance(obj, (nn.ModuleList, list)): layer_lists.append(obj)
    for layer_list in layer_lists:
        for lid, layer in enumerate(layer_list):
            for attr in ("attention", "attn", "self_attn"):
                if hasattr(layer, attr): _collect_attention_block(getattr(layer, attr), lid); break
        if weights: break
    return weights, names

__all__ = [
    "SpectronAttention",
    "LowRankSparseLinear",
    "SparsePruningCallback",
    "allocate_ranks_spectron",
    "inject_spectron",
    "spectral_entropy",
    "cumulative_entropy_fraction",
    "GLUE_TASK_KEYS",
    "get_num_labels_and_type",
    "preprocess",
    "gather_attention_weights",
]
