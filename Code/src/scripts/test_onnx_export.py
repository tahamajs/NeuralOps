# test_onnx_export.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def export_original_bert_to_onnx():
    print("🔄 Exporting original BERT to ONNX...")

    # Load the trained model with fixes
    model = AutoModelForSequenceClassification.from_pretrained(
        "./trained_bert",
        use_cache=False,
        return_dict=False
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Prepare dummy inputs
    dummy_input_text = ["This is a dummy sentence for ONNX export."]
    inputs = tokenizer(
        dummy_input_text,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    print("📤 Exporting with fixed settings...")

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "./original_bert_test.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        }
    )
    print("✅ Export successful with fixed settings.")

if __name__ == "__main__":
    export_original_bert_to_onnx()
