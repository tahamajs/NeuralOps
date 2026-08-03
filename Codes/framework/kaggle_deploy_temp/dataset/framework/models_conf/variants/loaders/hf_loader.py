from typing import Any
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from framework.models_conf.core_loaders import _safe_load
from .base import LOADER_REGISTRY, BaseLoader


@LOADER_REGISTRY.register("standard")
class HFStandardLoader(BaseLoader):
    def load(self, model_name, num_labels=2, cache_dir=None, **kwargs) -> Any:
        """Load using local disk cache first to avoid network hangs."""
        config = _safe_load(
            AutoConfig.from_pretrained,
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
        )
        tokenizer = _safe_load(
            AutoTokenizer.from_pretrained, model_name, cache_dir=cache_dir
        )
        model_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {"task", "attention_type", "attention_rank", "copy_weights", "rank"}
        }
        model = _safe_load(
            AutoModelForSequenceClassification.from_pretrained,
            model_name,
            config=config,
            cache_dir=cache_dir,
            **model_kwargs,
        )
        return model, tokenizer
