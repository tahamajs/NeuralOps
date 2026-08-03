import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from ..utils.layer_norm_fix import fix_layernorm_naming


def load_model(cfg, num_labels, device="auto") -> Any:
    """Load model using config and num_labels."""
    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, config=config
    )
    fix_layernorm_naming(model)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model
