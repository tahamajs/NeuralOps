"""
Unit tests for compression engines (SVD, SliceGPT, Kronecker-TT, Pruning, Quantization).
"""

import unittest
import torch
import torch.nn as nn
from framework.compression_engine.rotational_slicegpt_int4 import (
    RotationalSliceGPTINT4Compressor,
    RotationalSliceGPTCompressor,
)
from framework.compression_engine.kronecker_tt_lora import (
    KroneckerTTLoRACompressor,
    KroneckerTTLoRALayer,
)


class TestCompressionEngines(unittest.TestCase):
    """Test suite verifying compression engine algorithms, layer replacement, and execution."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 16
        self.in_dim = 64
        self.out_dim = 64
        self.dummy_x = torch.randn(self.batch_size, self.seq_len, self.in_dim)
        self.base_layer = nn.Linear(self.in_dim, self.out_dim)

    def test_rotational_slicegpt_int4(self):
        compressor = RotationalSliceGPTCompressor(r_ratio=0.5, quant_bits=4)
        compressed = compressor.compress_linear(self.base_layer)
        out = compressed(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

    def test_kronecker_tt_lora(self):
        compressor = KroneckerTTLoRACompressor(rank=8)
        compressed = compressor.compress_linear(self.base_layer)
        out = compressed(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))


if __name__ == "__main__":
    unittest.main()
