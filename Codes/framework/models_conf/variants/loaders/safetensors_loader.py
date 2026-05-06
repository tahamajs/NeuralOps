from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from .base import BaseLoader, LOADER_REGISTRY

@LOADER_REGISTRY.register("safetensors")
class SafeTensorsLoader(BaseLoader):
    def load(self, model_name, num_labels=2, cache_dir=None, **kwargs):
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, cache_dir=cache_dir, use_safetensors=True)
        return model, tokenizer