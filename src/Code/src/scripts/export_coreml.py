import torch
from soa.config import SOAConfig
from soa.model import SOABert
from soa.export_coreml import export_to_coreml

def main():
    cfg = SOAConfig(k_rank=64)
    model = SOABert(cfg)
    model.load_state_dict(torch.load("soa_migrated.pt", map_location="cpu"))
    export_to_coreml(model, input_shape=(1, 512), out_path="soa_bert.mlmodel")

if __name__ == "__main__":
    main()
