Module neuralops.compression_engine.huffman_coder
=================================================
Huffman Coder & Entropy Compression Engine for Weight and KV Cache Quantization.
Registered as 'huffman'. Production-hardened with vectorized encode/decode,
exact byte accounting, safe LoRA freezing, and robust tokenizer forward checks.

Functions
---------

`huffman_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Huffman compressor factory function.

Classes
-------

`HuffmanCoder()`
:   Vectorized lossless entropy coder for quantized weight and KV cache tensors.
    Supports continuous tensor quantization (e.g. 4-bit / 8-bit), symbol frequency modeling,
    binary prefix tree construction, numpy bitstream packing, and vectorized decoding.

    ### Static methods

    `build_tree(frequencies: dict[int, int]) ‑> neuralops.compression_engine.huffman_coder.HuffmanNode | None`
    :   Build Huffman binary prefix tree from symbol frequencies.

    `decode(packed_bytes: bytes, code_table: dict[int, str], metadata: dict[str, Any], device: torch.device | None = None) ‑> torch.Tensor`
    :   Vectorized decode bitstream back into reconstructed tensor.

    `encode(tensor: torch.Tensor, quantize_bits: int | None = 8) ‑> tuple[bytes, dict[int, str], dict[str, typing.Any]]`
    :   Vectorized encode input tensor into packed bitstream bytes and Huffman codebook.

    `generate_code_table(root: HuffmanNode | None, current_code: str = '', code_table: dict[int, str] | None = None) ‑> dict[int, str]`
    :   Traverse Huffman tree to generate prefix bitstring per symbol.

`HuffmanCompressor(model: nn.Module, quantize_bits: int = 8, max_error: float = 0.2, target_attention_only: bool = False, fallback_to_original: bool = True, cfg: Any | None = None, **kwargs: Any)`
:   HuffmanCompressor: Model-wide entropy compression profiler and simulator.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Profile and apply Huffman entropy coding across all linear layers.

`HuffmanNode(symbol: int | None = None, freq: int = 0, left: HuffmanNode | None = None, right: HuffmanNode | None = None)`
:   Node in the Huffman prefix tree.