from dataclasses import dataclass, field

from framework.utils.paths import get_glue_dir


@dataclass
class DataConfig:
    local_glue_path: str = str(get_glue_dir())
    max_seq_length: int = 128
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    num_proc: int = 4
    streaming: bool = False
    remove_columns: list[str] = field(default_factory=lambda: ["label"])
    split_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "train": "train",
            "validation": "validation",
            "validation_matched": "validation_matched",
            "validation_mismatched": "validation_mismatched",
        }
    )
