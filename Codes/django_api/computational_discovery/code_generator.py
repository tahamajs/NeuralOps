"""
Code variant generator – creates new versions of ML experiment code
by modifying hyperparameters, architecture choices, and training strategies.
"""

import random
import re


def generate_variant(
    base_code: str, previous_scores: list = None, temperature: float = 0.3
) -> dict:
    """
    Generate a new code variant from a base template.
    Uses simple heuristics – in production, this would call an LLM.
    """
    code = base_code

    # Possible modifications
    modifications = []

    # 1. Learning rate
    lr_options = [2e-5, 3e-5, 5e-5, 1e-4, 2e-4]
    new_lr = random.choice(lr_options)
    code = re.sub(r"learning_rate\s*=\s*[\d.e-]+", f"learning_rate = {new_lr}", code)
    modifications.append(f"Learning rate → {new_lr}")

    # 2. Batch size
    batch_options = [8, 16, 24, 32, 48, 64]
    new_batch = random.choice(batch_options)
    code = re.sub(r"batch_size\s*=\s*\d+", f"batch_size = {new_batch}", code)
    modifications.append(f"Batch size → {new_batch}")

    # 3. Number of epochs
    epoch_options = [3, 5, 7, 10, 15]
    new_epochs = random.choice(epoch_options)
    code = re.sub(r"epochs\s*=\s*\d+", f"epochs = {new_epochs}", code)
    modifications.append(f"Epochs → {new_epochs}")

    # 4. Optimizer
    optimizers = ["adamw", "adam", "sgd", "adafactor"]
    new_opt = random.choice(optimizers)
    code = re.sub(r'optimizer\s*=\s*["\']\w+["\']', f'optimizer = "{new_opt}"', code)
    modifications.append(f"Optimizer → {new_opt}")

    # 5. Dropout rate
    dropout_options = [0.0, 0.1, 0.2, 0.3]
    new_dropout = random.choice(dropout_options)
    code = re.sub(r"dropout\s*=\s*[\d.]+", f"dropout = {new_dropout}", code)
    modifications.append(f"Dropout → {new_dropout}")

    # 6. Compression rank (if applicable)
    rank_options = [8, 16, 32, 64, 128]
    new_rank = random.choice(rank_options)
    if "compression_rank" in code:
        code = re.sub(
            r"compression_rank\s*=\s*\d+", f"compression_rank = {new_rank}", code
        )
        modifications.append(f"Compression rank → {new_rank}")

    hyperparams = {
        "learning_rate": new_lr,
        "batch_size": new_batch,
        "epochs": new_epochs,
        "optimizer": new_opt,
        "dropout": new_dropout,
    }
    if "compression_rank" in code:
        hyperparams["compression_rank"] = new_rank

    return {
        "code": code,
        "modifications": modifications,
        "hyperparameters": hyperparams,
    }


def evaluate_variant(variant) -> dict:
    """
    Simulate evaluation of a code variant.
    In production, this would submit to the training pipeline.
    """
    # Simulated scoring based on hyperparameters
    hp = variant.hyperparameters
    base = 0.7
    # Slightly reward sensible hyperparameters
    if 2e-5 <= hp.get("learning_rate", 0) <= 5e-5:
        base += 0.05
    if hp.get("batch_size", 0) in (16, 32):
        base += 0.03
    if hp.get("dropout", 0) in (0.1, 0.2):
        base += 0.02
    if hp.get("optimizer", "") == "adamw":
        base += 0.03

    # Add some randomness
    import random as rnd

    base += rnd.uniform(-0.05, 0.05)

    return {
        "accuracy": round(min(0.99, base), 4),
        "f1": round(min(0.99, base - 0.01), 4),
        "loss": round(max(0.1, 1.0 - base + 0.1), 4),
    }
