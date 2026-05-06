# framework/data/augmentation.py
from framework.core.registry import Registry
import random

AUGMENTATION_REGISTRY = Registry("augmentation")

def random_mask_augment(dataset, mask_prob=0.15):
    def mask_tokens(example):
        input_ids = example["input_ids"]
        for i in range(len(input_ids)):
            if random.random() < mask_prob:
                input_ids[i] = 103  # [MASK] token id (BERT)
        example["input_ids"] = input_ids
        return example
    return dataset.map(mask_tokens)
AUGMENTATION_REGISTRY.register("random_mask")(random_mask_augment)

# simple back‑translation placeholder
def dummy_back_translation(dataset):
    return dataset
AUGMENTATION_REGISTRY.register("back_translation")(dummy_back_translation)