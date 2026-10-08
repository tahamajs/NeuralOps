"""Generate API documentation for neuralops public interfaces."""
import inspect

from neuralops.core import registry
from neuralops.inference import engine
from neuralops.compression_engine.quantization import gptq, awq, smoothquant
from neuralops.compression_engine.pruning import wanda


def main():
    print("# NeuralOps v1.1.0 API Reference\n")
    print("## Core Registries\n")
    print(f"- **ATTENTION_REGISTRY**: {len(registry.ATTENTION_REGISTRY)} registered modules.")
    print(f"- **COMPRESSOR_REGISTRY**: {len(registry.COMPRESSOR_REGISTRY)} registered modules.\n")

    print("## Key Modules\n")
    modules = [
        ("Inference Engine", engine),
        ("GPTQ Quantizer", gptq),
        ("AWQ Quantizer", awq),
        ("SmoothQuant", smoothquant),
        ("Wanda Pruner", wanda),
    ]

    for title, mod in modules:
        print(f"### {title}\n")
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name, None)
            doc = inspect.getdoc(obj) or "No docstring provided."
            doc_first = doc.strip().split("\n")[0]
            print(f"- **`{name}`**: {doc_first}")
        print()


if __name__ == "__main__":
    main()
