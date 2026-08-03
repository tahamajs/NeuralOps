from typing import Any
import torch

from .base import LOADER_REGISTRY, BaseLoader


@LOADER_REGISTRY.register("unsloth")
class UnslothLoader(BaseLoader):
    def load(self, model_name, num_labels=2, max_seq_length=2048, **kwargs) -> Any:
        """Load."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth not installed. Run: pip install unsloth")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        if not hasattr(model, "score"):
            model.score = torch.nn.Linear(model.config.hidden_size, num_labels)
            model.config.num_labels = num_labels
        return model, tokenizer
