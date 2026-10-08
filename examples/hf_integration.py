"""HuggingFace integration example."""
from neuralops.integrations.hf_automodel import AutoNeuralOpsModel

def main():
    model = AutoNeuralOpsModel.from_pretrained("dummy_model", attention_type="full_attention")
    print("HF AutoModel loaded successfully.")

if __name__ == "__main__":
    main()
