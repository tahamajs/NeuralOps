"""Architecture-family detection and common transformer model metadata."""
from __future__ import annotations

from typing import Any, Optional


FAMILIES = ("bert", "roberta", "gpt", "qwen", "t5", "ssm", "unknown")


def _config(obj: Any) -> Any:
    return getattr(obj, "config", obj)


def detect_family(model_or_config: Any) -> str:
    cfg = _config(model_or_config)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    cls = type(model_or_config).__name__.lower()
    if model_type.startswith("qwen") or "qwen" in cls:
        return "qwen"
    if model_type in {"bert", "distilbert", "electra", "deberta", "albert"}:
        return "bert"
    if model_type in {"roberta", "xlm-roberta"}:
        return "roberta"
    if model_type in {"t5", "mt5", "umt5", "bart", "pegasus"}:
        return "t5"
    if model_type in {"gpt", "gpt2", "gpt_neo", "gpt_neox", "llama", "mistral", "phi", "falcon"}:
        return "gpt"
    if any(x in model_type or x in cls for x in ("mamba", "ssm", "rwkv", "recurrent")):
        return "ssm"
    return "unknown"


class ModelAdapter:
    """Read-only architecture contract used by module wrappers."""

    def __init__(self, model_or_config: Any, family: Optional[str] = None):
        self.source = _config(model_or_config)
        self.family = family or detect_family(model_or_config)
        self.hidden_size = int(getattr(self.source, "hidden_size", getattr(self.source, "d_model", 0)))
        self.num_heads = int(getattr(self.source, "num_attention_heads",
                                     getattr(self.source, "num_heads",
                                             getattr(self.source, "num_decoder_attention_heads", 0))))
        self.head_dim = int(getattr(self.source, "head_dim", self.hidden_size // self.num_heads if self.num_heads else 0))
        kv_heads = getattr(
            self.source, "num_key_value_heads",
            getattr(self.source, "num_key_value_heads_decoder",
                    getattr(self.source, "num_decoder_attention_heads", self.num_heads or 0)),
        )
        self.num_key_value_heads = int(kv_heads or self.num_heads or 0)
        if self.num_key_value_heads <= 0:
            self.num_key_value_heads = self.num_heads
        self.num_hidden_layers = int(getattr(self.source, "num_hidden_layers", getattr(self.source, "num_layers", 0)))
        self.q_proj_out = self.num_heads * self.head_dim
        self.kv_proj_out = self.num_key_value_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_key_value_heads))
        self.projection_aliases = {
            "q": ("q_lin", "q_proj"), "k": ("k_lin", "k_proj"),
            "v": ("v_lin", "v_proj"), "o": ("out_lin", "o_proj"),
        }
        self.causal = self.family in {"gpt", "qwen", "ssm"}
        self.mask_rank = 4 if self.family in {"gpt", "qwen"} else (2 if self.family in {"bert", "roberta"} else 3)
        self.bidirectional = not self.causal
        self.position_embedding = "rotary" if self.family in {"gpt", "qwen"} else ("absolute" if self.family in {"bert", "roberta", "t5"} else "none")

    def projection(self, module: Any, kind: str) -> Any:
        legacy, native = self.projection_aliases[kind]
        return getattr(module, native, getattr(module, legacy, None))

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "hidden_size": self.hidden_size, "num_heads": self.num_heads,
                "num_key_value_heads": self.num_key_value_heads, "q_proj_out": self.q_proj_out,
                "kv_proj_out": self.kv_proj_out, "kv_repeat": self.kv_repeat,
                "head_dim": self.head_dim, "num_hidden_layers": self.num_hidden_layers,
                "causal": self.causal, "mask_rank": self.mask_rank,
                "position_embedding": self.position_embedding}
