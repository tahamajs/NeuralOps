import torch
import os
from .config import SOAConfig
from .model import SOABert

def export_onnx(rank=64, onnx_path="soa_bert.onnx"):
    # تنظیم متغیر محیطی برای اجبار به ذخیره داخلی وزن‌ها
    # این کار از ایجاد فایل‌های .data جداگانه که باعث خطا می‌شدند جلوگیری می‌کند
    os.environ["ONNX_EXTERNAL_DATA"] = "0"

    cfg = SOAConfig(k_rank=rank)
    model = SOABert(cfg)
    model.eval()

    # dummy inputs
    input_ids = torch.zeros(1, 128, dtype=torch.long)
    token_type_ids = torch.zeros(1, 128, dtype=torch.long)
    attention_mask = torch.ones(1, 128, dtype=torch.long)

    try:
        torch.onnx.export(
            model,
            (input_ids, token_type_ids, attention_mask),
            onnx_path,
            input_names=["input_ids", "token_type_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {1: "seq_len"},
                "token_type_ids": {1: "seq_len"},
                "attention_mask": {1: "seq_len"},
            },
            opset_version=18,
            do_constant_folding=True,
            # پارامتر save_as_external_data حذف شد چون در این نسخه پشتیبانی نمی‌شود
            # و با تنظیم متغیر محیطی بالا، رفتار صحیح اعمال می‌شود.
        )
        print(f"✅ ONNX exported successfully: {onnx_path}")
    except Exception as e:
        print(f"❌ Error during export: {e}")
        raise

import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--onnx_path", type=str, default="soa_bert.onnx")
    args = ap.parse_args()
    export_onnx(rank=args.rank, onnx_path=args.onnx_path)
