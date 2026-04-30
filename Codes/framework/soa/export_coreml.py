# framework/soa/export_coreml.py
import torch
from .config import SOAConfig
from .model import SOABert

def export_to_coreml(model=None, input_shape=(1, 512), out_path="soa_bert.mlmodel"):
    if model is None:
        cfg = SOAConfig(k_rank=64)
        model = SOABert(cfg)
        model.load_state_dict(torch.load("soa_migrated.pt", map_location="cpu"))
    model.eval()
    # Note: coremltools conversion would require additional code; this is a placeholder.
    print("CoreML export requires coremltools and a valid ONNX file.")
    # Example: ct.convert(...)