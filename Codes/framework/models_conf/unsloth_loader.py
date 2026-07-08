"""
Unsloth integration – loads and optionally replaces attention for fast training.
Requires: pip install unsloth
"""

import torch
from framework.core.attention_utils import inject_attention


def load_unsloth_model_with_attention(
    model_name: str,
    num_labels: int = 2,
    attention_type: str = "full",
    rank: int = 64,
    max_seq_length: int = 2048,
    device: str = "cuda",
    **kwargs,
) -> tuple:
    """
    Load a model with Unsloth acceleration and optionally replace attention.

    Args:
        model_name: HuggingFace model name or path (supports Llama, Mistral, etc.)
        num_labels: number of output classes.
        attention_type: attention registry key to inject.
        rank: compression rank.
        max_seq_length: maximum sequence length for the model.
        device: target device.

    Returns:
        model, tokenizer
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth is not installed. Run: pip install unsloth")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    if attention_type != "full":
        # Add classification head if missing
        if not hasattr(model, "score"):
            model.score = torch.nn.Linear(model.config.hidden_size, num_labels).to(
                torch.float16
            )
            model.config.num_labels = num_labels

        inject_attention(model, attention_type, rank=rank, copy_weights=True)

    return model, tokenizer
