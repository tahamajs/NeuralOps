"""HuggingFace Hub utilities for NeuralOps models."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
import torch


__all__ = [
    "scan_for_secrets",
    "generate_model_card",
    "push_model_to_hub",
    "load_model_from_hub",
    "list_org_models",
]

SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{34,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),
]


def scan_for_secrets(directory_path: str | Path) -> list[str]:
    """Scan files in a directory for leaked tokens or secret API keys."""
    findings: list[str] = []
    p = Path(directory_path)
    for f in p.rglob("*"):
        if f.is_file() and not f.name.endswith((".bin", ".pt", ".safetensors", ".png", ".jpg")):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                for pattern in SECRET_PATTERNS:
                    if pattern.search(content):
                        findings.append(f"Potential secret in {f.name}")
            except Exception:
                pass
    return findings


def generate_model_card(
    model_name: str,
    config: Any,
    metrics: dict[str, Any] | None = None,
    license_type: str = "mit",
) -> str:
    """Generate a markdown model card for HuggingFace Hub repository."""
    card = [
        "---",
        f"language: en",
        f"license: {license_type}",
        f"library_name: neuralops",
        f"tags:",
        f"- subquadratic-attention",
        f"- model-compression",
        f"- pytorch",
        f"- transformers",
        "---",
        "",
        f"# {model_name}",
        "",
        "## Architecture Details",
        f"- **Model Type**: `{getattr(config, 'model_type', 'neuralops')}`",
        f"- **Hidden Size**: `{getattr(config, 'hidden_size', 'N/A')}`",
        f"- **Layers**: `{getattr(config, 'num_hidden_layers', 'N/A')}`",
        f"- **Heads**: `{getattr(config, 'num_attention_heads', 'N/A')}`",
        f"- **Attention Type**: `{getattr(config, 'attention_type', 'N/A')}`",
        f"- **Compression**: `{getattr(config, 'compression_type', 'None')}`",
        "",
        "## Usage",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "import neuralops.models",
        "",
        f'model = AutoModelForCausalLM.from_pretrained("{model_name}")',
        "```",
        "",
    ]
    if metrics:
        card.extend([
            "## Evaluation Metrics",
            "| Metric | Value |",
            "|---|---|",
        ])
        for k, v in metrics.items():
            card.append(f"| {k} | {v} |")
        card.append("")

    return "\n".join(card)


def push_model_to_hub(
    model: Any,
    repo_id: str,
    tokenizer: Any = None,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Upload NeuralOps model",
) -> str:
    """Safely push a NeuralOps model to Hugging Face Hub."""
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN must be provided as parameter or environment variable.")

    # Temporarily save and secret-scan
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp_dir)

        # Generate model card
        card_content = generate_model_card(repo_id, model.config)
        (Path(tmp_dir) / "README.md").write_text(card_content)

        leaks = scan_for_secrets(tmp_dir)
        if leaks:
            raise ValueError(f"Aborting push to hub. Secrets detected: {leaks}")

        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            commit_message=commit_message,
        )

    return f"https://huggingface.co/{repo_id}"


def load_model_from_hub(repo_id: str, **kwargs: Any) -> Any:
    """Load a NeuralOps model directly from HuggingFace Hub."""
    from neuralops.models.modeling import NeuralOpsForCausalLM
    return NeuralOpsForCausalLM.from_pretrained(repo_id, **kwargs)


def list_org_models(org: str = "neuralops-hub") -> list[str]:
    """List all available models in the specified organization."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        models = api.list_models(author=org)
        return [m.id for m in models]
    except Exception:
        return []
