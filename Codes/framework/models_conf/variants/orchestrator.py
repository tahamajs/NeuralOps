"""
Orchestrator that applies multiple variants in the correct order:
1. Loader
2. Quantization
3. PEFT (adapters)
4. Compression
5. Export
6. Serving (optional)
"""

from .compression import COMPRESSOR_REGISTRY
from .export import EXPORTER_REGISTRY
from .loaders import LOADER_REGISTRY
from .peft import PEFT_REGISTRY
from .quantization import QUANTIZER_REGISTRY
from .serving import SERVER_REGISTRY


def apply_variants(
    model_name: str,
    num_labels: int = 2,
    loader: str = "standard",
    quantizer: str = None,
    peft: str = None,
    compressor: str = None,
    exporter: str = None,
    server: str = None,
    **kwargs,
):
    # 1. Load
    loader_cls = LOADER_REGISTRY.resolve(loader)
    model, tokenizer = loader_cls().load(model_name, num_labels=num_labels, **kwargs)

    # 2. Quantize
    if quantizer:
        q_cls = QUANTIZER_REGISTRY.resolve(quantizer)
        model = q_cls().apply(model, **kwargs)

    # 3. PEFT
    if peft:
        p_cls = PEFT_REGISTRY.resolve(peft)
        model = p_cls().apply(model, **kwargs)

    # 4. Compress
    if compressor:
        c_cls = COMPRESSOR_REGISTRY.resolve(compressor)
        model = c_cls().apply(model, **kwargs)

    # 5. Export (optional)
    exported_path = None
    if exporter:
        e_cls = EXPORTER_REGISTRY.resolve(exporter)
        exported_path = e_cls().export(
            model, kwargs.get("output_path", "model_export"), **kwargs
        )

    # 6. Serve (optional)
    server_instance = None
    if server:
        s_cls = SERVER_REGISTRY.resolve(server)
        server_instance = s_cls().serve(model, **kwargs)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "exported_path": exported_path,
        "server": server_instance,
    }
