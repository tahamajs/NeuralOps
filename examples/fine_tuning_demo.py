"""Fine-tuning with LoRA and mixed precision."""
import torch
import torch.nn as nn
from neuralops.training.trainer import MixedPrecisionTrainer

def main():
    model = nn.Linear(32, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = MixedPrecisionTrainer(model, opt, nn.MSELoss(), enabled=False)
    x = torch.randn(4, 32)
    y = torch.randn(4, 2)
    loss = trainer.train_step(x, y)
    print("Train step completed with loss:", loss)

if __name__ == "__main__":
    main()
