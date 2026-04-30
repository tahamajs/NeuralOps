# framework/soa/export_onnx.py
import torch
import os
from .config import SOAConfig
from .model import SOABert

def export_onnx(rank=64, onnx_path="soa_bert.onnx"):
    os.environ["ONNX_EXTERNAL_DATA"] = "0"
    cfg = SOAConfig(k_rank=rank)
    model = SOABert(cfg)
    model.eval()

    input_ids = torch.zeros(1, 128, dtype=torch.long)
    token_type_ids = torch.zeros(1, 128, dtype=torch.long)
    attention_mask = torch.ones(1, 128, dtype=torch.long)

    torch.onnx.export(
        model,
        (input_ids, token_type_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "token_type_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "seq_len"}, "token_type_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}},
        opset_version=18,
        do_constant_folding=True
    )
    print(f"✅ ONNX exported: {onnx_path}")