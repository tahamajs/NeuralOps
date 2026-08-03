"""Huffman coding for weight compression. Standalone utility."""
from typing import Any

import heapq

import torch


class HuffmanCoder:
    @staticmethod
    def encode(weights: torch.Tensor) -> Any:
        """Return (encoded_bits, code_map) for a weight tensor."""
        flat = weights.view(-1).tolist()
        freq = {}
        for w in flat:
            freq[w] = freq.get(w, 0) + 1

        heap = [[freq, [sym, ""]] for sym, freq in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = "0" + pair[1]
            for pair in hi[1:]:
                pair[1] = "1" + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        code_map = dict(heap[0][1:])
        encoded = [code_map[w] for w in flat]
        return encoded, code_map

    @staticmethod
    def decode(encoded, code_map) -> Any:
        """Decode a list of bit strings back to weight values."""
        reverse_map = {v: k for k, v in code_map.items()}
        return [reverse_map[bits] for bits in encoded]
