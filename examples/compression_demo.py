"""Model compression demonstration."""
import torch
import torch.nn as nn
from neuralops.compression_engine.quantization.gptq import quantize_model_gptq

def main():
    model = nn.Sequential(nn.Linear(64, 64))
    calib = [torch.randn(16, 64) for _ in range(4)]
    quantized = quantize_model_gptq(model, calib, bits=4)
    print("Model quantized with GPTQ.")

if __name__ == "__main__":
    main()
