# soa/export_onnx.py
import torch
import os
from .config import SOAConfig
from .model import SOABert

def export_onnx(model, rank, onnx_path="soa_bert.onnx"):
    # Set environment variable to avoid external data files
    os.environ["ONNX_EXTERNAL_DATA"] = "0"
    
    model.eval()
    dummy_input_ids = torch.zeros(1, 128, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(1, 128, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_token_type_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "token_type_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "token_type_ids": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
        },
        opset_version=17, # Opset 17 is well-supported for M1
        do_constant_folding=True,
    )
    print(f"  ✅ Exported ONNX model to {onnx_path}")