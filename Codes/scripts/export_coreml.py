import torch
import coremltools as ct
from soa.config import SOAConfig
from soa.model import SOABert

def export_to_coreml(rank=64):
    cfg = SOAConfig(k_rank=rank)
    model = SOABert(cfg)
    model.load_state_dict(torch.load(f"soa_bert_rank_{rank}.pt", map_location="cpu"))
    model.eval()
    example_input = (torch.randint(0, 30522, (1, 128)), torch.zeros(1, 128, dtype=torch.long), torch.ones(1, 128, dtype=torch.long))
    traced = torch.jit.trace(model, example_input)
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(shape=(1, 128)), ct.TensorType(shape=(1, 128)), ct.TensorType(shape=(1, 128))])
    mlmodel.save(f"soa_bert_rank_{rank}.mlmodel")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=64)
    args = ap.parse_args()
    export_to_coreml(args.rank)