"""
Huffman Coder & Entropy Compression Engine for Weight and KV Cache Quantization.
Registered as 'huffman'.
"""
from __future__ import annotations

import heapq
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value"}


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
        """
        flat = tensor.detach().float().view(-1)
        numel = flat.numel()
        if numel == 0:
            return b"", {}, {"total_elements": 0, "entropy": 0.0, "compressed_bytes": 0}

        min_val = flat.min().item()
        max_val = flat.max().item()

        if quantize_bits is not None and quantize_bits < 16:
            num_levels = 2 ** int(quantize_bits)
            scale = (max_val - min_val) / max(num_levels - 1, 1)
            scale = max(scale, 1e-12)
            quantized = torch.clamp(
                torch.round((flat - min_val) / scale), 0, num_levels - 1
            ).long()
        else:
            scale = 1.0
            quantized = flat.long()

        symbols, counts = torch.unique(quantized, return_counts=True)
        frequencies = {int(s.item()): int(c.item()) for s, c in zip(symbols, counts)}

        root = HuffmanCoder.build_tree(frequencies)
        code_table = HuffmanCoder.generate_code_table(root)

        probs = [c / numel for c in frequencies.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0.0)

        # Packed bitstream
        bit_chunks = [code_table.get(int(v.item()), "0") for v in quantized]
        bitstream_str = "".join(bit_chunks)
        pad_len = (8 - len(bitstream_str) % 8) % 8
        padded_bitstream = bitstream_str + "0" * pad_len

        byte_array = bytearray()
        for i in range(0, len(padded_bitstream), 8):
            byte_array.append(int(padded_bitstream[i : i + 8], 2))

        metadata = {
            "scale": scale,
            "min_val": min_val,
            "orig_shape": list(tensor.shape),
            "total_elements": numel,
            "entropy": round(entropy, 3),
            "pad_len": pad_len,
            "compressed_bytes": len(byte_array),
        }
        return bytes(byte_array), code_table, metadata

    @staticmethod
    def decode(
        packed_bytes: bytes,
        code_table: Dict[int, str],
        metadata: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Decode bitstream back into reconstructed tensor."""
        total_elements = metadata.get("total_elements", 0)
        if total_elements == 0:
            return torch.empty(metadata.get("orig_shape", [0]), device=device)

        rev_table = {v: k for k, v in code_table.items()}
        pad_len = metadata.get("pad_len", 0)

        bit_list = [f"{b:08b}" for b in packed_bytes]
        bitstream = "".join(bit_list)
        if pad_len > 0:
            bitstream = bitstream[:-pad_len]

        decoded_symbols: List[int] = []
        cur_buf = ""
        for bit in bitstream:
            cur_buf += bit
            if cur_buf in rev_table:
                decoded_symbols.append(rev_table[cur_buf])
                cur_buf = ""
                if len(decoded_symbols) == total_elements:
                    break

        # De-quantize
        scale = metadata.get("scale", 1.0)
        min_val = metadata.get("min_val", 0.0)
        shape = metadata.get("orig_shape", [total_elements])

        t_symbols = torch.tensor(decoded_symbols, dtype=torch.float32, device=device)
        reconstructed = t_symbols * scale + min_val
        return reconstructed.reshape(shape)


@COMPRESSOR_REGISTRY.register("huffman")
class HuffmanCompressor(BaseCompressor):
    """
    HuffmanCompressor: Model-wide entropy compression profiler and simulator.
    """

    def __init__(
        self,
        model: nn.Module,
        quantize_bits: int = 8,
        max_error: float = 0.20,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="huffman", **kwargs)
        cfg = cfg or kwargs.get("comp_config", None)
        self.cfg = cfg

        self.quantize_bits = int(_cfg_get(cfg, "huffman_quantize_bits", _cfg_get(cfg, "quantize_bits", _cfg_get(cfg, "bits", quantize_bits))))
        self.max_error = float(_cfg_get(cfg, "huffman_max_error", _cfg_get(cfg, "max_error", max_error)))
        self.target_attention_only = bool(
            _cfg_get(cfg, "huffman_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(cfg, "huffman_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
            return False

        if self.target_attention_only:
            if not any(attn_kw in name_lower for attn_kw in _ATTN_NAMES):
                return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False
        if not should_compress_linear(self.model, name):
            return False
        return True

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Profile and apply Huffman entropy coding across all linear layers."""
        first_p = next(self.model.parameters(), None)
        _model_device = first_p.device if first_p is not None else torch.device("cpu")

        total_orig_bytes = 0
        total_comp_bytes = 0
        total_entropy = 0.0

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set[int] = set()

        # Freeze skipped non-target layers if target_attention_only is enabled
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
                    continue
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        for name, module in self.model.named_modules():
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data
            orig_shape = raw_w.shape
            orig_device = raw_w.device
            orig_dtype = raw_w.dtype

            W_orig = raw_w.t().float() if is_conv1d else raw_w.float()
            norm_w = torch.norm(W_orig, p="fro").item()

            try:
                packed_b, code_map, meta = HuffmanCoder.encode(W_orig, quantize_bits=self.quantize_bits)
                W_rec = HuffmanCoder.decode(packed_b, code_map, meta, device=orig_device).to(dtype=torch.float32)

                if norm_w < 1e-12:
                    err = 0.0
                else:
                    err = float(torch.norm(W_orig - W_rec, p="fro").item() / norm_w)

                self.layer_errors[name] = err

                if err > self.max_error:
                    logger.warning("Huffman error %.4f on %s exceeded max_error %.4f", err, name, self.max_error)
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False
                        total_orig_bytes += raw_w.numel() * 2
                        total_comp_bytes += raw_w.numel() * 2
                        continue

                # Apply reconstructed weight
                if is_conv1d:
                    res_w = W_rec.t().to(dtype=orig_dtype, device=orig_device)
                else:
                    res_w = W_rec.to(dtype=orig_dtype, device=orig_device)

                assert res_w.shape == orig_shape, f"Shape mismatch on {name}: {res_w.shape} vs {orig_shape}"
                target_mod.weight.data.copy_(res_w)

                # Preserve LoRA adapter trainability
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    if hasattr(module.lora_A, "weight") and module.lora_A.weight is not None:
                        module.lora_A.weight.requires_grad = True
                    if hasattr(module.lora_B, "weight") and module.lora_B.weight is not None:
                        module.lora_B.weight.requires_grad = True

                self.compressed_layers.append(name)
                total_orig_bytes += raw_w.numel() * 2
                total_comp_bytes += meta["compressed_bytes"]
                total_entropy += meta["entropy"]

            except Exception as e:
                logger.debug("Huffman exception on %s: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_orig_bytes += raw_w.numel() * 2
                total_comp_bytes += raw_w.numel() * 2

        # Ensure all parameters on target device
        self.model.to(device=_model_device)

        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next((m for m in self.model.modules() if isinstance(m, nn.Linear)), None)
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception:
            forward_check_success = False

        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / max(1, len(self.layer_errors))) if self.layer_errors else 0.0
        stats["quantize_bits"] = self.quantize_bits
        stats["avg_entropy_bits"] = round(total_entropy / max(len(self.compressed_layers), 1), 2)
        stats["total_compressed_bytes"] = total_comp_bytes
        stats["huffman_compression_ratio"] = round(
            total_orig_bytes / max(total_comp_bytes, 1), 2
        )
        stats["theoretical_compression_ratio"] = stats["huffman_compression_ratio"]
        stats["num_compressed"] = len(self.compressed_layers)
        stats["num_fallback"] = len(self.fallback_layers)
        stats["num_skipped"] = len(self.skipped_layers)
        stats["compressed_layers"] = self.compressed_layers
        stats["fallback_layers"] = self.fallback_layers
        stats["skipped_layers"] = self.skipped_layers
        stats["avg_reconstruction_error"] = avg_err
        stats["forward_check_success"] = forward_check_success
        return self.model, stats


@COMPRESSOR_REGISTRY.register("huffman")
def huffman_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Huffman compressor factory function."""
    bits = kwargs.pop("quantize_bits", kwargs.pop("bits", _cfg_get(comp_config, "huffman_bits", _cfg_get(comp_config, "quantize_bits", 8))))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "huffman_max_error", _cfg_get(comp_config, "max_error", 0.20)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "huffman_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "huffman_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)),
    )
    compressor = HuffmanCompressor(
        model,
        quantize_bits=bits,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = ["HuffmanNode", "HuffmanCoder", "HuffmanCompressor", "huffman_factory"]
