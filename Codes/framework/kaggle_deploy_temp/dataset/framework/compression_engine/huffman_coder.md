# Huffman Coding for Weight Compression

**Standalone utility**  
*Compress weight tensors via lossless variable‑length encoding, reducing storage for models with many repeated weight values.*

---

## Overview

Huffman coding is a classic lossless data compression algorithm that assigns shorter bit sequences to more frequent symbols and longer codes to rare ones. In the context of neural network compression, it can be applied to the raw weight values of a trained model to reduce their storage footprint without any information loss. This utility provides a simple Python implementation of Huffman encoding and decoding for PyTorch tensors. It is not a compressor registered in the DistilBERT framework but a standalone helper that can be used independently to compress weight lists.

The class `HuffmanCoder` contains two static methods:
- `encode(weights: torch.Tensor)` → returns the encoded bit‑strings and the code mapping.
- `decode(encoded, code_map)` → reconstructs the original list of weight values (as a Python list of floats).

This utility is particularly useful after applying lossy compression techniques (e.g., pruning, quantization) that reduce the number of distinct weight values; Huffman coding can then exploit the non‑uniform distribution of those discrete values to further shrink the model’s storage.

---

## Methodology

### 1. Frequency‑Based Code Assignment

Given a flattened list of weight values (floats), the encoder first counts how often each distinct value appears. It then builds a **Huffman tree**:

- Each distinct value becomes a leaf node with weight = its frequency.
- The two nodes with the smallest frequencies are repeatedly merged into a new parent node whose frequency is the sum, until only one node remains (the root).
- During the merge, a `'0'` is prepended to the codes of the left child’s symbols, and `'1'` to the right child’s symbols.

After the tree is built, every distinct weight value is assigned a binary code (string of `'0'`/`'1'`). More frequent values receive shorter codes.

### 2. Encoding

The input weight tensor is flattened, and each entry is replaced by its corresponding Huffman code from the mapping. The result is a list of strings (the encoded bits) and the mapping dictionary.

### 3. Decoding

Given the list of encoded strings and the mapping dictionary, the decoder reverses the mapping (from code to weight value) and reconstructs the original list of values (as Python floats). The shape of the original tensor must be stored separately if one wishes to restore the tensor.

### 4. Compression Efficiency

The compression ratio depends on the entropy of the weight distribution. If weights are highly concentrated on a few values (e.g., after aggressive quantization to 2–4 bits), Huffman coding can achieve significant savings. For continuous floating‑point weights with many unique values, the overhead of the mapping may outweigh the gains. The method is lossless, so the original weight values are perfectly recovered.

---

## Implementation Details

### HuffmanCoder Class

```python
class HuffmanCoder:
    @staticmethod
    def encode(weights: torch.Tensor):
        # 1. count frequencies of each distinct weight value
        # 2. build Huffman tree using a min‑heap (heapq)
        # 3. derive code map (value -> binary string)
        # 4. encode the flattened list using the code map
        return encoded, code_map

    @staticmethod
    def decode(encoded, code_map):
        reverse_map = {v: k for k, v in code_map.items()}
        return [reverse_map[bits] for bits in encoded]
```

**Note**: The code contains a stray line `self.dropout = nn.Dropout(config.dropout)` which appears to be a copy‑paste error from a different module; it has no effect inside the `encode` method because `self` is not defined (the method is static). It is ignored.

### Usage Example

```python
import torch
from huffman_coder import HuffmanCoder

# Example: quantized weights (many repeated values)
weights = torch.tensor([1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0])
encoded, code_map = HuffmanCoder.encode(weights)
print("Encoded bits:", encoded)
# Output: list of bit strings, e.g., ['0', '0', '10', '11', '0', '10', '11', '11']

decoded = HuffmanCoder.decode(encoded, code_map)
print("Decoded values:", decoded)
# [1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0]
```

To apply to a real model, one would iterate over parameters, flatten, encode, store the code map, and later decode.

---

## Discussion

### Advantages

- **Lossless**: Perfect reconstruction of weight values.
- **Exploits redundancy**: After quantization or pruning, the number of distinct weight values drops, making Huffman coding highly effective.
- **Simple and standalone**: No dependencies beyond Python standard library.

### Limitations

- **Overhead for continuous distributions**: If the weight tensor has too many unique values, the code map itself can be large, negating the compression.
- **Bit‑string representation**: The encoded data is stored as strings of `'0'` and `'1'`, which are not memory‑efficient in Python. For practical compression, one would pack the bits into a byte array.
- **Not integrated**: This utility is not a registered compressor; it does not modify the model architecture. It is meant to be used as a post‑processing step to compress serialized weight data.
- **No shape information**: Decoding only returns a flat list; the tensor shape must be stored separately to reconstruct the original tensor.

### When to Use

- After applying a quantization method that reduces weights to a small set of discrete levels.
- As a final step before storing a model checkpoint to disk, achieving additional lossless compression.
- For experimenting with information‑theoretic limits of weight compression.

---

## Conclusion

The Huffman coding utility provides a classic lossless compression technique adapted for neural network weight tensors. Its `encode`/`decode` interface makes it easy to apply to any flat list of weights. While it is not a full‑fledged compressor integrated into the DistilBERT pipeline, it serves as a valuable companion tool for squeezing out the last bits of redundancy after other compression methods have been applied. Future improvements could include direct bit‑packing and integration with a model‑serialization format for seamless compression of checkpoints.