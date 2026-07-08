"""
Simple data augmentation registry (random mask, back‑translation placeholder).
"""

import random

from framework.core.registry import Registry

AUGMENTATION_REGISTRY = Registry("augmentation")


@AUGMENTATION_REGISTRY.register("random_mask")
def random_mask_augment(dataset, mask_prob=0.15):
    def mask_tokens(example):
        input_ids = example["input_ids"]
        for i in range(len(input_ids)):
            if random.random() < mask_prob:
                input_ids[i] = 103  # [MASK] token id (BERT)
        example["input_ids"] = input_ids
        return example

    return dataset.map(mask_tokens)


@AUGMENTATION_REGISTRY.register("back_translation")
def dummy_back_translation(dataset):
    # Placeholder – replace with actual back‑translation logic
    return dataset
