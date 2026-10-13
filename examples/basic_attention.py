"""
Example: Basic Attention Replacement (Surgery)

Shows how to:
  1. Load a pretrained transformer model (DistilBERT).
  2. Replace standard Multi-Head Attention with a subquadratic variant (SVD Linear Attention).
  3. Execute forward pass with inputs and inspect output logits.

Requires:
  - neuralops installed (pip install neuralops)
  - Optional: CUDA GPU (falls back to CPU)
  - Optional: Triton (falls back to PyTorch)

Run:
    python examples/basic_attention.py

Expected output:
    Successfully replaced attention and produced valid output logits.
"""
import torch
from transformers import AutoModelForSequenceClassification
import neuralops


def main() -> None:
    print("Loading pretrained DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

    print("Replacing attention layers with 'svd_linear_attention' (rank=128)...")
    model = neuralops.replace_attention(model, "svd_linear_attention", rank=128)

    # Run forward pass
    inputs = torch.randint(0, 1000, (2, 64))
    with torch.no_grad():
        outputs = model(inputs)

    print(f"Forward pass successful! Logits shape: {outputs.logits.shape}")
    assert outputs.logits.shape == (2, 2)
    print("✅ Successfully replaced attention and produced valid output logits.")


if __name__ == "__main__":
    main()
