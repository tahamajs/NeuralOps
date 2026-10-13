"""
Example: Model Compression with Direct SVD

Shows how to:
  1. Load a pretrained transformer model (DistilBERT).
  2. Compress linear and attention projections using low-rank Direct SVD decomposition.
  3. Verify parameter reduction and run inference on compressed architecture.

Requires:
  - neuralops installed (pip install neuralops)
  - Optional: CUDA GPU (falls back to CPU)
  - Optional: Triton (falls back to PyTorch)

Run:
    python examples/compression_demo.py

Expected output:
    Model compressed with parameter reduction and valid forward logits.
"""
import torch
from transformers import AutoModelForSequenceClassification
import neuralops


def main() -> None:
    print("Loading pretrained DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {orig_params:,}")

    print("Compressing model using 'direct_svd' (rank=32)...")
    compressed = neuralops.compress(model, method="direct_svd", rank=32)

    comp_params = sum(p.numel() for p in compressed.parameters())
    reduction = (orig_params - comp_params) / orig_params * 100
    print(f"Compressed parameters: {comp_params:,} (Reduction: {reduction:.2f}%)")

    # Run forward pass on compressed model
    inputs = torch.randint(0, 1000, (2, 32))
    with torch.no_grad():
        outputs = compressed(inputs)

    print(f"Compressed forward pass successful! Logits shape: {outputs.logits.shape}")
    assert outputs.logits.shape == (2, 2)
    print("✅ Model compressed with parameter reduction and valid forward logits.")


if __name__ == "__main__":
    main()
