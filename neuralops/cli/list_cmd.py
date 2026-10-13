"""`list` command."""
import json


def cmd_list(args):
    from neuralops.core.registry import (
        ATTENTION_REGISTRY, COMPRESSOR_REGISTRY,
    )
    import neuralops.attentions  # noqa: F401
    import neuralops.compression_engine  # noqa: F401

    attn = sorted(ATTENTION_REGISTRY.keys())
    comp = sorted(COMPRESSOR_REGISTRY.keys())

    if args.json:
        data = {}
        if args.type in ("attention", "all"):
            data["attention"] = attn
        if args.type in ("compressor", "all"):
            data["compressor"] = comp
        print(json.dumps(data, indent=2))
        return 0

    if args.type in ("attention", "all"):
        print(f"Attention ({len(attn)}):")
        for k in attn:
            print(f"  {k}")
    if args.type in ("compressor", "all"):
        print(f"Compressor ({len(comp)}):")
        for k in comp:
            print(f"  {k}")
    return 0
