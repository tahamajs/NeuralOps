# soa/export_coreml.py
import torch
import coremltools as ct
from .config import SOAConfig
from .model import SOABert

def export_to_coreml(model, input_shape=(1, 128), out_path="soa_bert.mlpackage"):
    """
    Exports the SOA model to Core ML format.
    Note: Requires the model to be in evaluation mode.
    """
    model.eval()
    
    # Create dummy inputs for tracing
    dummy_input_ids = torch.zeros(input_shape, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(input_shape, dtype=torch.long)
    dummy_attention_mask = torch.ones(input_shape, dtype=torch.long)
    
    # Trace the model
    traced_model = torch.jit.trace(model, (dummy_input_ids, dummy_token_type_ids, dummy_attention_mask))
    
    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=dummy_input_ids.shape),
            ct.TensorType(name="token_type_ids", shape=dummy_token_type_ids.shape),
            ct.TensorType(name="attention_mask", shape=dummy_attention_mask.shape),
        ],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL, # Use CPU, GPU, and Neural Engine
    )
    
    # Save the model
    mlmodel.save(out_path)
    print(f"✅ Core ML model saved to {out_path}")