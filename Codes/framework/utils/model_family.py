"""Architecture-family detection and common transformer model metadata.

Bug fixes (2026-09-25):
- Fix MF-01: Added "gpt2" (and gpt_neo, gpt_neox, llama, mistral, phi, falcon)
  to the causal family set. Previously ModelAdapter(family="gpt2") reported
  causal=False, causing causal masks to be dropped and recovery to fail on
  GPT-2. This was a critical correctness bug.
- Fix MF-02: Added GPT_FAMILIES and CAUSAL_FAMILIES frozensets to guarantee
  consistent behavior across causal, mask_rank, and position_embedding
  decisions.
- Fix MF-03: "gpt2" and other GPT variants now correctly report
  position_embedding="absolute" (learned position embeddings).
- Fix MF-04: mask_rank for all causal families is now 4.
- Fix MF-05: Added family alias normalization in ModelAdapter.__init__ so
  that any of {"gpt", "gpt2", "gpt_neo", "gpt_neox", "llama", "mistral",
  "phi", "falcon"} map to the canonical internal family "gpt", while
  preserving the raw family name in raw_family for logging.
"""
from __future__ import annotations

from typing import Any, Optional


FAMILIES = ("bert", "roberta", "gpt", "qwen", "qwen3_5", "t5", "ssm", "unknown")


# ---------------------------------------------------------------------------
# Canonical family sets (single source of truth)
# ---------------------------------------------------------------------------
# All model types that use causal attention. This is the authoritative list.
CAUSAL_FAMILIES = frozenset({
    "gpt", "gpt2", "gpt_neo", "gpt_neox",
    "llama", "mistral", "phi", "falcon",
    "qwen", "qwen3_5",
    "ssm",
})

# Causal decoder families whose input uses learned absolute position embeddings.
GPT_ABSOLUTE_POSITION_FAMILIES = frozenset({
    "gpt", "gpt2", "gpt_neo", "gpt_neox",
    "phi", "falcon",
})

# Causal decoder families using rotary position embeddings.
ROTARY_FAMILIES = frozenset({
    "qwen", "qwen3_5", "llama", "mistral",
})

# Bidirectional encoder families (for mask_rank=2).
BIDIRECTIONAL_FAMILIES = frozenset({"bert", "roberta"})

# Family normalization aliases: map raw family names to canonical family names.
FAMILY_ALIASES = {
    "gpt2": "gpt",
    "gpt_neo": "gpt",
    "gpt_neox": "gpt",
    "llama": "gpt",
    "mistral": "gpt",
    "phi": "gpt",
    "falcon": "gpt",
}


def _config(obj: Any) -> Any:
    cfg = getattr(obj, "config", obj)
    if hasattr(cfg, "text_config"):
        return cfg.text_config
    return cfg


def detect_family(model_or_config: Any) -> str:
    """Detect the model family from a model or config object."""
    cfg = _config(model_or_config)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    cls = type(model_or_config).__name__.lower()

    if any(x in model_type or x in cls for x in ("qwen3_5", "qwen3.5", "qwen35", "qwen3")):
        return "qwen3_5"
    if model_type.startswith("qwen") or "qwen" in cls:
        return "qwen"

    if model_type in {"bert", "distilbert", "electra", "deberta", "albert"}:
        return "bert"
    if model_type in {"roberta", "xlm-roberta"}:
        return "roberta"
    if model_type in {"t5", "mt5", "umt5", "bart", "pegasus"}:
        return "t5"

    # All GPT-style causal decoders map to canonical "gpt" family.
    if model_type in {
        "gpt", "gpt2", "gpt_neo", "gpt_neox",
        "llama", "mistral", "phi", "falcon",
    }:
        return "gpt"

    if any(x in model_type or x in cls for x in ("mamba", "ssm", "rwkv", "recurrent")):
        return "ssm"

    return "unknown"


class ModelAdapter:
    """Read-only architecture contract used by module wrappers.

    Fix MF-05: `family` is normalized to a canonical internal name (e.g. "gpt2"
    becomes "gpt"), while the original name is preserved in `raw_family` for
    logging. Downstream code that inspects `adapter.family` will always see a
    canonical value; code that needs the raw value should use `raw_family`.
    """

    def __init__(self, model_or_config: Any, family: Optional[str] = None):
        self.source = _config(model_or_config)
        raw_family = family or detect_family(model_or_config)
        # Normalize aliases to canonical internal family.
        self.raw_family = raw_family
        self.family = FAMILY_ALIASES.get(raw_family, raw_family)

        self.hidden_size = int(
            getattr(self.source, "hidden_size",
                    getattr(self.source, "d_model", 0)) or 0
        )
        heads = getattr(
            self.source, "num_attention_heads",
            getattr(self.source, "num_heads",
                    getattr(self.source, "num_decoder_attention_heads", 0)),
        )
        self.num_heads = int(heads or 0)
        h_dim = getattr(self.source, "head_dim", None)
        self.head_dim = int(
            h_dim if h_dim is not None
            else (self.hidden_size // self.num_heads if self.num_heads else 0)
        )
        kv_heads = getattr(
            self.source, "num_key_value_heads",
            getattr(self.source, "num_key_value_heads_decoder",
                    getattr(self.source, "num_decoder_attention_heads",
                            self.num_heads or 0)),
        )
        self.num_key_value_heads = int(kv_heads or self.num_heads or 0)
        if self.num_key_value_heads <= 0:
            self.num_key_value_heads = self.num_heads
        self.num_hidden_layers = int(
            getattr(self.source, "num_hidden_layers",
                    getattr(self.source, "num_layers", 0))
        )
        self.q_proj_out = self.num_heads * self.head_dim
        self.kv_proj_out = self.num_key_value_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_key_value_heads))
        self.projection_aliases = {
            "q": ("q_lin", "q_proj"),
            "k": ("k_lin", "k_proj"),
            "v": ("v_lin", "v_proj"),
            "o": ("out_lin", "o_proj"),
        }

        # ------------------------------------------------------------------
        # Fix MF-01/MF-02/MF-03/MF-04: Consistent causal-family handling.
        # ------------------------------------------------------------------
        # A family is causal if either:
        #   (a) its canonical family is in CAUSAL_FAMILIES, OR
        #   (b) its raw family is in CAUSAL_FAMILIES (defense in depth).
        self.causal = (self.family in CAUSAL_FAMILIES) or (raw_family in CAUSAL_FAMILIES)

        # mask_rank: 4 for causal decoders, 2 for bidirectional encoders, 3 otherwise.
        if self.causal:
            self.mask_rank = 4
        elif self.family in BIDIRECTIONAL_FAMILIES:
            self.mask_rank = 2
        else:
            self.mask_rank = 3

        self.bidirectional = not self.causal

        # position_embedding: rotary for modern decoders, absolute for BERT/GPT-2,
        # relative for T5, none for unknown.
        if self.family in ROTARY_FAMILIES or raw_family in ROTARY_FAMILIES:
            self.position_embedding = "rotary"
        elif (self.family in GPT_ABSOLUTE_POSITION_FAMILIES
              or raw_family in GPT_ABSOLUTE_POSITION_FAMILIES
              or self.family in {"bert", "roberta"}):
            self.position_embedding = "absolute"
        elif self.family == "t5":
            self.position_embedding = "relative"
        else:
            self.position_embedding = "none"

        self.supports_gqa = (self.num_key_value_heads < self.num_heads) or self.family == "qwen3_5"
        self.supports_delta_net = self.family == "qwen3_5"
        self.supports_moe = self.family == "qwen3_5"

    def projection(self, module: Any, kind: str) -> Any:
        legacy, native = self.projection_aliases[kind]
        return getattr(module, native, getattr(module, legacy, None))

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "raw_family": self.raw_family,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "q_proj_out": self.q_proj_out,
            "kv_proj_out": self.kv_proj_out,
            "kv_repeat": self.kv_repeat,
            "head_dim": self.head_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "causal": self.causal,
            "mask_rank": self.mask_rank,
            "position_embedding": self.position_embedding,
            "supports_gqa": self.supports_gqa,
            "supports_delta_net": self.supports_delta_net,
            "supports_moe": self.supports_moe,
        }