"""
Model‑agnostic architecture helpers.
Provides unified access to layers, attention, FFN, embeddings, classifiers
across all common HuggingFace architectures.
"""

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def detect_architecture(model: nn.Module) -> str:
    """Return canonical architecture name."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "") if config else ""
    name = model.__class__.__name__.lower()
    if "distilbert" in model_type or "distilbert" in name:
        return "distilbert"
    if "roberta" in model_type or "roberta" in name:
        return "roberta"
    if "bert" in model_type or "bert" in name:
        return "bert"
    if "gpt2" in model_type or "gpt2" in name:
        return "gpt2"
    if "qwen" in model_type or "qwen" in name:
        return "qwen"
    if "llama" in model_type:
        return "llama"
    if "mistral" in model_type:
        return "mistral"
    if "gemma" in model_type:
        return "gemma"
    if "phi" in model_type:
        return "phi"
    if "t5" in model_type:
        return "t5"
    return "unknown"


def get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Return a flat list of transformer blocks."""
    # Llama / Gemma / Phi / Mistral / Falcon
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # GPT‑2 / OPT / GPT‑J
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # BERT / RoBERTa / DistilBert / ALBERT
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return list(model.encoder.layer)
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        return list(model.bert.encoder.layer)
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        return list(model.roberta.encoder.layer)
    if hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
        return list(model.distilbert.transformer.layer)
    # T5 encoder / decoder
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        return list(model.encoder.block)
    if hasattr(model, "decoder") and hasattr(model.decoder, "block"):
        return list(model.decoder.block)
    raise ValueError("Could not locate transformer layers automatically.")


def get_attention_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return (name, module) for all leaf attention sub-modules across any transformer architecture.

    Applies strict path-prefix deduplication so parent composite containers are never
    returned alongside their inner leaf attention blocks, preventing accidental deletion
    of LayerNorm, output projections, and residual streams.
    """
    raw_candidates: list[tuple[str, nn.Module]] = []
    seen_names = set()

    for name, module in model.named_modules():
        if not name:
            continue
        if name in seen_names:
            continue

        class_name = module.__class__.__name__.lower()
        if "universalattentionwrapper" in class_name or "adaptivespan" in class_name:
            continue
        if name.endswith(".inner") or name.endswith("._wrapped_attention"):
            continue
        is_output_submodule = (
            name.endswith(".output")
            or name.endswith("_output")
            or "selfoutput" in class_name
        )
        if is_output_submodule:
            continue

        # Heuristic 1: Module class name indicates attention or linear-recurrent block
        if any(
            x in class_name
            for x in [
                "attention",
                "selfattn",
                "selfattention",
                "crossattn",
                "deltanet",
                "gated_deltanet",
                "gatedattention",
            ]
        ):
            raw_candidates.append((name, module))
            seen_names.add(name)
            continue

        # Heuristic 2: Attribute-based detection (modules holding projections)
        if (
            hasattr(module, "q_proj")
            or hasattr(module, "k_proj")
            or hasattr(module, "q_lin")
            or hasattr(module, "query")
            or hasattr(module, "c_attn")
            or hasattr(module, "qk_proj")
        ):
            # Avoid leaf nn.Linear layers that might happen to match attributes
            if not isinstance(
                module, (nn.Linear, nn.LayerNorm, nn.Dropout, nn.Embedding)
            ):
                raw_candidates.append((name, module))
                seen_names.add(name)

    # Apply strict path-prefix deduplication:
    # If module A's path is a strict prefix of module B's path (e.g. A="...attention", B="...attention.self"),
    # A is an outer container holding B. We keep only the deepest/leaf modules (B) and drop A.
    all_cand_names = [name for name, _ in raw_candidates]
    leaf_modules: list[tuple[str, nn.Module]] = []

    for name, module in raw_candidates:
        is_parent = any(
            other != name and other.startswith(name + ".") for other in all_cand_names
        )
        if not is_parent:
            leaf_modules.append((name, module))

    # Expected count sanity validation based on model configuration
    cfg = getattr(model, "config", None)
    if cfg is not None:
        enc_layers = getattr(
            cfg,
            "num_hidden_layers",
            getattr(cfg, "num_layers", getattr(cfg, "n_layer", None)),
        )
        is_enc_dec = getattr(cfg, "is_encoder_decoder", False)
        dec_layers = getattr(cfg, "num_decoder_layers", enc_layers) if is_enc_dec else 0

        if enc_layers is not None:
            expected_min = (enc_layers + 2 * dec_layers) if is_enc_dec else enc_layers
            if len(leaf_modules) != expected_min:
                logger.warning(
                    "get_attention_modules detected %d modules, expected approximately %d "
                    "(layers=%s, is_encoder_decoder=%s). Detected: %s",
                    len(leaf_modules),
                    expected_min,
                    enc_layers,
                    is_enc_dec,
                    [n for n, _ in leaf_modules],
                )

    return leaf_modules


def get_ffn_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return (name, module) for all feed‑forward blocks."""
    ffn_modules = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if any(x in class_name for x in ["mlp", "ffn", "feedforward"]):
            ffn_modules.append((name, module))
            continue
        if (
            hasattr(module, "gate_proj")
            or hasattr(module, "up_proj")
            or hasattr(module, "intermediate")
        ):
            ffn_modules.append((name, module))
    return ffn_modules


def get_embedding_module(model: nn.Module) -> nn.Module | None:
    """Return the input word embedding module."""
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return module
    return None


def get_classifier_head(model: nn.Module) -> nn.Module | None:
    """Return the final classification layer."""
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "score"):
        return model.score
    if hasattr(model, "pre_classifier"):
        return model.pre_classifier, model.classifier
    return None
