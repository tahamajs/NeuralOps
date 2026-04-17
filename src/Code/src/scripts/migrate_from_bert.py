import torch
from transformers import BertModel
from soa.config import SOAConfig
from soa.model import SOABert
from soa.svd_migration import compute_global_basis_from_bert, migrate_bert_to_soa

def main():
    bert = BertModel.from_pretrained("bert-base-uncased")
    basis, k = compute_global_basis_from_bert(bert, k=64)

    cfg = SOAConfig(k_rank=k)
    soa = SOABert(cfg)

    soa = migrate_bert_to_soa(bert, soa, basis)
    torch.save(soa.state_dict(), "soa_migrated.pt")
    print("Saved migrated SOA model: soa_migrated.pt")

if __name__ == "__main__":
    main()
