"""
Huffman Coder & Entropy Compression Engine for Weight and KV Cache Quantization.
Registered as 'huffman'.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class HuffmanNode:
    """Node in the Huffman prefix tree."""

    def __init__(
        self,
        symbol: Optional[int] = None,
        freq: int = 0,
        left: Optional[HuffmanNode] = None,
        right: Optional[HuffmanNode] = None,
    ) -> None:
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: HuffmanNode) -> bool:
        return self.freq < other.freq


class HuffmanCoder:
    """
    Huffman lossless entropy coder for quantized weight and KV cache tensors.
    Supports continuous tensor quantization (e.g. 4-bit / 8-bit), symbol frequency modeling,
    binary prefix tree construction, bitstream packing, and lossless decoding.
    """

    @staticmethod
    def build_tree(frequencies: Dict[int, int]) -> Optional[HuffmanNode]:
        """Build Huffman binary prefix tree from symbol frequencies."""
        if not frequencies:
            return None

        heap: List[HuffmanNode] = [HuffmanNode(sym, freq) for sym, freq in frequencies.items()]
        heapq.heapify(heap)

        if len(heap) == 1:
            root = HuffmanNode(freq=heap[0].freq, left=heap[0])
            return root

        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = HuffmanNode(freq=node1.freq + node2.freq, left=node1, right=node2)
            heapq.heappush(heap, merged)

        return heap[0]

    @staticmethod
    def generate_code_table(
        root: Optional[HuffmanNode], current_code: str = "", code_table: Optional[Dict[int, str]] = None
    ) -> Dict[int, str]:
        """Traverse Huffman tree to generate prefix bitstring per symbol."""
        if code_table is None:
            code_table = {}
        if root is None:
            return code_table

        if root.symbol is not None:
            code_table[root.symbol] = current_code if current_code else "0"
            return code_table

        if root.left is not None:
            HuffmanCoder.generate_code_table(root.left, current_code + "0", code_table)
        if root.right is not None:
            HuffmanCoder.generate_code_table(root.right, current_code + "1", code_table)

        return code_table

    @staticmethod
    def encode(
        tensor: torch.Tensor,
        quantize_bits: Optional[int] = 8,
    ) -> Tuple[bytes, Dict[int, str], Dict[str, Any]]:
        """
        Encode input tensor into packed bitstream bytes and Huffman codebook.
        Returns:
            packed_bytes: Packed bytes array of variable-length codes.
            code_map: Dictionary mapping integer symbols to bitstrings.
            metadata: Shape, quantization scale/min, total bits, entropy, efficiency.
        """
        flat = tensor.detach().float().view(-1)
        numel = flat.numel()
        if numel == 0:
            return b"", {}, {"total_elements": 0, "entropy": 0.0}

        min_val = flat.min().item()
        max_val = flat.max().item()

        # Step 1: Discretize if quantize_bits is specified
        if quantize_bits is not None:
            q_levels = (2**quantize_bits) - 1
            scale = max((max_val - min_val) / float(max(q_levels, 1)), 1e-8)
            discrete_symbols = torch.clamp(torch.round((flat - min_val) / scale), 0, q_levels).long().tolist()
        else:
            scale = 1.0
            discrete_symbols = flat.long().tolist()

        # Step 2: Frequency analysis
        freq_map: Dict[int, int] = {}
        for sym in discrete_symbols:
            freq_map[sym] = freq_map.get(sym, 0) + 1

        # Step 3: Build tree & generate code table
        root = HuffmanCoder.build_tree(freq_map)
        code_map = HuffmanCoder.generate_code_table(root)

        # Step 4: Bitstream encoding & byte packing
        bitstring_list = [code_map[sym] for sym in discrete_symbols]
        total_bitstring = "".join(bitstring_list)
        total_bits = len(total_bitstring)

        # Pack bitstring into bytearray
        packed_bytes = bytearray()
        for i in range(0, total_bits, 8):
            byte_chunk = total_bitstring[i : i + 8]
            if len(byte_chunk) < 8:
                byte_chunk = byte_chunk.ljust(8, "0")
            packed_bytes.append(int(byte_chunk, 2))

        # Step 5: Information-theoretic stats
        entropy = 0.0
        avg_code_len = 0.0
        for sym, count in freq_map.items():
            prob = count / float(numel)
            entropy -= prob * math.log2(max(prob, 1e-12))
            avg_code_len += prob * len(code_map[sym])

        efficiency = (entropy / max(avg_code_len, 1e-6)) * 100.0
        fp16_bytes = numel * 2
        compressed_bytes = len(packed_bytes) + len(code_map) * 4  # packed data + codebook overhead
        compression_ratio = fp16_bytes / max(compressed_bytes, 1)

        metadata = {
            "orig_shape": list(tensor.shape),
            "total_elements": numel,
            "quantize_bits": quantize_bits,
            "min_val": min_val,
            "scale": scale,
            "total_bits": total_bits,
            "entropy": round(entropy, 3),
            "avg_code_length": round(avg_code_len, 3),
            "coding_efficiency_pct": round(efficiency, 2),
            "compressed_bytes": compressed_bytes,
            "compression_ratio": round(compression_ratio, 2),
        }
        return bytes(packed_bytes), code_map, metadata

    @staticmethod
    def decode(
        packed_bytes: bytes,
        code_map: Dict[int, str],
        metadata: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Decode packed bytes back to reconstructed floating point tensor."""
        total_elements = metadata.get("total_elements", 0)
        if total_elements == 0:
            return torch.empty(0, device=device)

        total_bits = metadata.get("total_bits", len(packed_bytes) * 8)
        # Unpack bytes to bitstring
        bitstring = "".join(f"{b:08b}" for b in packed_bytes)[:total_bits]

        # Invert code map
        reverse_map = {code: sym for sym, code in code_map.items()}

        decoded_symbols: List[int] = []
        cur_code = ""
        for bit in bitstring:
            cur_code += bit
            if cur_code in reverse_map:
                decoded_symbols.append(reverse_map[cur_code])
                cur_code = ""
                if len(decoded_symbols) == total_elements:
                    break

        # De-quantize
        scale = metadata.get("scale", 1.0)
        min_val = metadata.get("min_val", 0.0)
        shape = metadata.get("orig_shape", [total_elements])

        t_symbols = torch.tensor(decoded_symbols, dtype=torch.float32, device=device)
        reconstructed = t_symbols * scale + min_val
        return reconstructed.reshape(shape)


class HuffmanCompressor(BaseCompressor):
    """
    HuffmanCompressor: Model-wide entropy compression profiler and simulator.

    Args:
        model: PyTorch model.
        quantize_bits: Pre-quantization bit-width before Huffman encoding (default 8).
    """

    def __init__(
        self,
        model: nn.Module,
        quantize_bits: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="huffman", **kwargs)
        self.quantize_bits = int(quantize_bits)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Profile and apply Huffman entropy coding across all linear layers."""
        total_orig_bytes = 0
        total_comp_bytes = 0
        total_entropy = 0.0
        num_layers = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                packed_b, code_map, meta = HuffmanCoder.encode(W, quantize_bits=self.quantize_bits)
                total_orig_bytes += W.numel() * 2
                total_comp_bytes += meta["compressed_bytes"]
                total_entropy += meta["entropy"]
                num_layers += 1

                # Reconstruct simulated weights
                W_rec = HuffmanCoder.decode(packed_b, code_map, meta, device=W.device)
                module.weight.data = W_rec.to(dtype=module.weight.dtype)

        stats = self.get_compression_stats()
        stats["quantize_bits"] = self.quantize_bits
        stats["avg_entropy_bits"] = round(total_entropy / max(num_layers, 1), 2)
        stats["total_compressed_bytes"] = total_comp_bytes
        stats["huffman_compression_ratio"] = round(
            total_orig_bytes / max(total_comp_bytes, 1), 2
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("huffman")
def huffman_factory(model: nn.Module, comp_config: Any = None) -> nn.Module:
    """Huffman compressor factory function."""
    bits = getattr(comp_config, "huffman_bits", 8) if comp_config else 8
    compressor = HuffmanCompressor(model, quantize_bits=bits)
    compressed_model, _ = compressor.compress()
    return compressed_model
