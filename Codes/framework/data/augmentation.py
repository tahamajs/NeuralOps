from framework.core.registry import Registry

AUGMENTATION_REGISTRY = Registry("augmentation")

# Text augmentation examples (token-level)
def simple_noise_augment(dataset, noise_prob=0.05):
    import random
    def add_noise(example):
        tokens = example["input_ids"]
        for i in range(len(tokens)):
            if random.random() < noise_prob:
                tokens[i] = random.randint(0, 30000)  # generic vocab id
        example["input_ids"] = tokens
        return example
    return dataset.map(add_noise)

AUGMENTATION_REGISTRY.register("simple_noise")(simple_noise_augment)