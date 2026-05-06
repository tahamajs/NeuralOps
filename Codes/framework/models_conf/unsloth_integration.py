"""
Load a model using Unsloth and swap attention.
Requires: pip install unsloth
"""

import torch
from .base_model_wrapper import replace_attention_in_model


def load_unsloth_model_with_attention(model_name, num_labels, attention_type="full", rank=64, device="cuda"):
    """
    Load a model with Unsloth acceleration and optionally replace attention.

    Args:
        model_name: HuggingFace model name or path (supports Llama, Mistral, etc.)
        num_labels: number of output classes.
        attention_type: attention registry key to inject.
        rank: compression rank.
        device: target device.

    Returns:
        model, tokenizer
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth is not installed. Run: pip install unsloth")

    # Load the model and tokenizer with Unsloth's optimized loader
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,          # adjust as needed
        dtype=torch.float16,
        load_in_4bit=True,            # 4‑bit quantization for efficiency
    )

    if attention_type != "full":
        # Add a classification head if it doesn't exist
        if not hasattr(model, 'score'):
            model.score = torch.nn.Linear(model.config.hidden_size, num_labels).to(torch.float16)
            model.config.num_labels = num_labels

        model = replace_attention_in_model(model, attention_type, rank=rank)

    return model, tokenizer