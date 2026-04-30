# framework/compression/huffman_coder.py
import torch
import torch.nn as nn
from heapq import heappush, heappop

class HuffmanCoder:
    @staticmethod
    def encode(weights):
        # Build frequency map
        freq = {}
        for w in weights.view(-1).tolist():
            freq[w] = freq.get(w, 0) + 1
        # Build Huffman tree
        heap = [[freq, [sym, ""]] for sym, freq in freq.items()]
        while len(heap) > 1:
            lo = heappop(heap)
            hi = heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        code_map = dict(heap[0][1:])
        encoded = [code_map[w] for w in weights.view(-1).tolist()]
        return encoded, code_map

    @staticmethod
    def decode(encoded, code_map):
        reverse_map = {v: k for k, v in code_map.items()}
        decoded = [reverse_map[bits] for bits in encoded]
        return decoded