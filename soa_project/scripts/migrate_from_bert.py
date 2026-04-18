import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from soa.config import SOAConfig
from soa.model import SOABert
from soa.svd_migration import compute_global_basis, migrate_bert_to_soa

def main():
    print("🔵 Starting BERT to SOA migration...")
    # 1. Load pre-trained BERT
    print("Loading pre-trained 'bert-base-uncased'...")
    try:
        # Using BertModel ensures we get the base model without classification head
        bert_model = BertModel.from_pretrained("bert-base-uncased")
    except Exception as e:
        print(f"❌ Failed to load BERT model: {e}")
        return

    # 2. Configuration
    cfg = SOAConfig(k_rank=64)  # Adjust rank here
    
    # 3. Compute Global Basis
    print(f"Computing global basis U with rank={cfg.k_rank}...")
    basis = compute_global_basis(bert_model, cfg.k_rank)
    
    # 4. Initialize SOA model and migrate weights
    print("Initializing SOA model and migrating weights...")
    soa_model = SOABert(cfg)
    soa_model = migrate_bert_to_soa(bert_model, soa_model, basis)
    
    # 5. Save the migrated model
    output_path = f"soa_bert_rank_{cfg.k_rank}.pt"
    torch.save(soa_model.state_dict(), output_path)
    print(f"✅ Migration successful! Saved SOA model to '{output_path}'")

if __name__ == "__main__":
    main()