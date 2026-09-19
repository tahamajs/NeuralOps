import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk

ckpt_path = r"D:\svd_project\results\unified_sweep\roberta\finetuned_model.pt"
print(f"=== Inspecting Checkpoint: {ckpt_path} ===")
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Checkpoint keys:", list(ckpt.keys()))
for k, v in ckpt.items():
    print(f"  {k}: shape={v.shape}, mean={v.float().mean().item():.5f}, std={v.float().std().item():.5f}")

tokenizer = AutoTokenizer.from_pretrained(r"D:\svd_project\models\roberta-base")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

base = AutoModelForSequenceClassification.from_pretrained(
    r"D:\svd_project\models\roberta-base", 
    num_labels=2, 
    ignore_mismatched_sizes=True
)
print(f"Model config num_labels: {base.config.num_labels}")
print(f"Model config vocab_size: {base.config.vocab_size}")

missing, unexpected = base.load_state_dict(ckpt, strict=False)
print(f"Loaded checkpoint with missing keys ({len(missing)}): {missing[:5]}...")
print(f"Unexpected keys ({len(unexpected)}): {unexpected}")

base.eval()
base.to("cuda")

val = load_from_disk(r"D:\svd_project\Codes\framework\data\glue_data\sst2")["validation"]
print(f"Validation dataset size: {len(val)}")

first_10_logits = []
for i in range(10):
    inp = tokenizer(val[i]["sentence"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = base(**inp)
        first_10_logits.append(out.logits.cpu().numpy()[0])

print("\nFirst 10 sample logits [class 0, class 1]:")
for i, l in enumerate(first_10_logits):
    print(f"Sample {i} (true label={val[i]['label']}): logits={l}, pred={l.argmax()}")
