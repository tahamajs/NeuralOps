"""
Example: Hugging Face AutoModel Integration

Shows how to:
  1. Configure a subquadratic transformer with AutoConfig (e.g. cosformer or gla).
  2. Instantiate an AutoModelForCausalLM model directly from configuration.
  3. Generate autoregressive tokens using the Hugging Face generation API.

Requires:
  - neuralops installed (pip install neuralops)
  - Optional: CUDA GPU (falls back to CPU)
  - Optional: Triton (falls back to PyTorch)

Run:
    python examples/hf_integration.py

Expected output:
    Generated autoregressive token sequence shape matching requested parameters.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import neuralops.models


def main() -> None:
    print("Building Hugging Face configuration for 'cosformer' attention...")
    config = AutoConfig.for_model(
        "neuralops",
        attention_type="cosformer",
        vocab_size=1000,
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        max_position_embeddings=512,
    )

    print("Instantiating AutoModelForCausalLM from configuration...")
    model = AutoModelForCausalLM.from_config(config)

    prompt = torch.randint(0, 1000, (1, 8))
    print(f"Generating completions for prompt of length {prompt.shape[1]}...")
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=16)

    print(f"Generated sequence shape: {out.shape}")
    assert out.shape == (1, 24)
    print("✅ Generated autoregressive token sequence shape matching requested parameters.")


if __name__ == "__main__":
    main()
