from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"
    num_labels: int = 2
    attention_type: str = "full"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    cache_dir: str = "./cache"
    fix_layernorm_naming: bool = True
    variants: list[str] = field(default_factory=list)
