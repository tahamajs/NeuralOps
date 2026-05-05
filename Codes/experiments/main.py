# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))

# import json
# import torch
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from framework.compression.compress_utils import compress_bert_true_svd
# from framework.soa.config import SOAConfig
# from framework.soa.model import SOABert
# from framework.soa.migration import compute_global_basis_from_bert, migrate_bert_to_soa
# from framework.utils.device import get_device
# from framework.training.finetune import fine_tune_soa_model, train_bert_model
# from config import config

# def main():
#     device = get_device()
#     print(f"🚀 Running on {device}")

#     # 1. Train original BERT (or load pre‑trained)
#     bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     # ... (load data, train for 1 epoch – simplified)
#     # For brevity, assume we have a trained model saved.

#     # 2. Compute shared basis from BERT
#     basis, k = compute_global_basis_from_bert(bert_model.bert, config.svd_rank)

#     # 3. Create SOA model and migrate weights
#     soa_cfg = SOAConfig(k_rank=k)
#     soa_model = SOABert(soa_cfg).to(device)
#     soa_model = migrate_bert_to_soa(bert_model.bert, soa_model, basis)

#     # 4. Fine‑tune SOA on local dataset
#     train_loader, test_loader = load_local_data(tokenizer)  # (helper not shown)
#     fine_tune_soa_model(soa_model, train_loader, test_loader, device, epochs=3)

#     # 5. Save and export to ONNX
#     torch.save(soa_model.state_dict(), f"soa_bert_rank_{k}.pt")
#     export_soa_to_onnx(soa_model, k)

#     print("✅ Pipeline complete.")

# if __name__ == "__main__":
#     main()