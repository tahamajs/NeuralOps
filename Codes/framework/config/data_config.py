from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class DataConfig:
    local_glue_path: str = "/datasets/glue"
    max_seq_length: int = 128
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    num_proc: int = 4
    streaming: bool = False
    remove_columns: List[str] = field(default_factory=lambda: ["label"])
    split_mapping: Dict[str, str] = field(default_factory=lambda: {
        "train": "train",
        "validation": "validation",
        "validation_matched": "validation_matched",
        "validation_mismatched": "validation_mismatched",
    })