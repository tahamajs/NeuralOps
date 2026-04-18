import os
import torch
import argparse
from transformers import BertModel
from soa.config import SOAConfig
from soa.model import SOABert
from soa.svd_migration import compute_global_basis, migrate_bert_to_soa
from soa.export_onnx import export_onnx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_rank", type=int, default=32)
    parser.add_argument("--max_rank", type=int, default=128)
    parser.add_argument("--step", type=int, default=16)
    args = parser.parse_args()

    print("🚀 Starting rank sweep...")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    
    ranks = list(range(args.min_rank, args.max_rank + 1, args.step))
    for rank in ranks:
        print(f"\n--- Processing Rank: {rank} ---")
        
        # 1. Compute basis for this rank
        basis = compute_global_basis(bert_model, rank)
        
        # 2. Create and migrate SOA model
        cfg = SOAConfig(k_rank=rank)
        soa_model = SOABert(cfg)
        soa_model = migrate_bert_to_soa(bert_model, soa_model, basis)
        
        # 3. Save PyTorch model
        pt_path = f"soa_bert_rank_{rank}.pt"
        torch.save(soa_model.state_dict(), pt_path)
        print(f"  ✅ Saved PyTorch model to {pt_path}")
        
        # 4. Export to ONNX
        onnx_path = f"soa_bert_rank_{rank}.onnx"
        os.environ["ONNX_EXTERNAL_DATA"] = "0"
        export_onnx(soa_model, rank, onnx_path)
        print(f"  ✅ Exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    main()