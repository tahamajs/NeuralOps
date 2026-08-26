import os
import glob
import ast
import json

PROJECT_ROOT = "/Users/tahamajs/Documents/uni/Project"
METADATA_DIR = os.path.join(PROJECT_ROOT, "module_metadata")
os.makedirs(METADATA_DIR, exist_ok=True)

def extract_metadata_from_file(filepath, category):
    filename = os.path.basename(filepath)
    mod_name = os.path.splitext(filename)[0]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        source_code = f.read()
        
    try:
        tree = ast.parse(source_code)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None
        
    metadata = {
        "module_name": mod_name,
        "category": category, # "attention" or "compression"
        "filepath": filepath,
        "class_name": None,
        "base_classes": [],
        "docstring": "",
        "init_params": [],
        "forward_ops": [],
        "tensor_transforms": [],
        "math_primitives": [],
        "complexity": {
            "time_complexity": "O(N d)",
            "space_complexity": "O(N d)",
            "scaling_exponent": 1.0,
            "description": "Linear / efficient scaling"
        },
        "family": "linear",
        "accent_color": "#38bdf8", # default blue
        "key_mechanisms": []
    }
    
    # Extract class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if metadata["class_name"] is None or any(k in node.name for k in ["Attention", "Compress", "SVD", "Layer", "Mamba", "Linear", "MoE", "Kernel", "Quant"]):
                metadata["class_name"] = node.name
                metadata["base_classes"] = [b.id if isinstance(b, ast.Name) else ast.unparse(b) for b in node.bases]
                doc = ast.get_docstring(node)
                if doc:
                    metadata["docstring"] = doc[:300]
                    
                # Inspect __init__
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        for arg in item.args.args:
                            if arg.arg != "self" and arg.arg not in metadata["init_params"]:
                                metadata["init_params"].append(arg.arg)
                                
                    elif isinstance(item, ast.FunctionDef) and item.name in ["forward", "compress", "decompose", "apply"]:
                        for subnode in ast.walk(item):
                            if isinstance(subnode, ast.Call):
                                func_str = ""
                                if isinstance(subnode.func, ast.Attribute):
                                    func_str = subnode.func.attr
                                elif isinstance(subnode.func, ast.Name):
                                    func_str = subnode.func.id
                                
                                func_lower = func_str.lower()
                                if func_lower in ["matmul", "bmm", "einsum", "softmax", "sigmoid", "gelu", "relu", "silu", "tanh", "cumsum", "fft", "rfft", "irfft", "svd", "pca", "topk", "conv1d", "conv2d", "linear", "layernorm", "rmsnorm", "dropout", "qr", "lu", "chunk", "split", "view", "reshape", "transpose", "permute"]:
                                    if func_str not in metadata["forward_ops"]:
                                        metadata["forward_ops"].append(func_str)
                                        
                            elif isinstance(subnode, ast.BinOp):
                                if isinstance(subnode.op, ast.MatMult):
                                    if "matmul (@)" not in metadata["math_primitives"]:
                                        metadata["math_primitives"].append("matmul (@)")
                                elif isinstance(subnode.op, ast.Mult):
                                    if "elementwise (*)" not in metadata["math_primitives"]:
                                        metadata["math_primitives"].append("elementwise (*)")

    if not metadata["class_name"]:
        metadata["class_name"] = "".join(word.capitalize() for word in mod_name.split("_"))

    # Determine Family, Accent Color, and Mathematical Complexity Profile
    full_text = (source_code + " " + metadata["class_name"] + " " + mod_name).lower()
    
    if any(k in full_text for k in ["mamba", "ssm", "state_space", "recurrent", "s4", "s5", "rwkv", "gla", "lru", "h3", "selective_scan"]):
        metadata["family"] = "ssm"
        metadata["accent_color"] = "#10b981" # Green
        metadata["complexity"]["time_complexity"] = "O(N d)"
        metadata["complexity"]["space_complexity"] = "O(N d + d_s)"
        metadata["complexity"]["scaling_exponent"] = 1.0
        metadata["complexity"]["description"] = "Recurrent / Selective State-Space O(N)"
        metadata["key_mechanisms"] = ["Discretized State Matrix A", "Input-Dependent Step Delta", "Associative Prefix Scan", "Hidden State Recurrence"]
    elif any(k in full_text for k in ["moe", "expert", "mixture", "routing", "switch", "bam_umoe", "topk", "gating"]):
        metadata["family"] = "moe"
        metadata["accent_color"] = "#f59e0b" # Orange
        metadata["complexity"]["time_complexity"] = "O(N d k / E)"
        metadata["complexity"]["space_complexity"] = "O(E d^2)"
        metadata["complexity"]["scaling_exponent"] = 1.15
        metadata["complexity"]["description"] = "Sparse Mixture-of-Experts Dynamic Routing"
        metadata["key_mechanisms"] = ["Softmax Router Gating", "Top-K Expert Dispatch", "Specialized FFN / Projections", "Weighted Expert Aggregation"]
    elif any(k in full_text for k in ["sparse", "prun", "slicegpt", "wanda", "magnitude", "mask", "sparse_attention", "local_attention", "bigbird", "longformer", "block_sparse"]):
        metadata["family"] = "sparse"
        metadata["accent_color"] = "#ef4444" # Red
        metadata["complexity"]["time_complexity"] = "O(N \\sqrt{N} d)"
        metadata["complexity"]["space_complexity"] = "O(N w)"
        metadata["complexity"]["scaling_exponent"] = 1.3
        metadata["complexity"]["description"] = "Structural / Dynamic Sparsity O(N sqrt(N))"
        metadata["key_mechanisms"] = ["Sparsity Thresholding", "Local Sliding Window", "Global Token Indirection", "Dynamic Block Pruning"]
    elif any(k in full_text for k in ["kernel", "performer", "cosformer", "rff", "random_feature", "fourier", "spectral", "fast_attention", "fnet"]):
        metadata["family"] = "kernel"
        metadata["accent_color"] = "#a855f7" # Purple
        metadata["complexity"]["time_complexity"] = "O(N m d)"
        metadata["complexity"]["space_complexity"] = "O(N m)"
        metadata["complexity"]["scaling_exponent"] = 1.0
        metadata["complexity"]["description"] = "Kernelized Feature Map / Spectral Operator"
        metadata["key_mechanisms"] = ["Positive Orthogonal RFF", "Kernel Feature Map phi(x)", "Associative Re-ordering", "Spectral Fourier Mode Decomp"]
    elif any(k in full_text for k in ["svd", "lowrank", "lora", "tensor_train", "tt_", "nmf", "matrix_decomposition", "factoriz", "cur", "dora", "qlora"]):
        metadata["family"] = "linear"
        metadata["accent_color"] = "#38bdf8" # Blue
        metadata["complexity"]["time_complexity"] = "O(N r d)"
        metadata["complexity"]["space_complexity"] = "O(r(N + d))"
        metadata["complexity"]["scaling_exponent"] = 1.0
        metadata["complexity"]["description"] = "Low-Rank SVD / Factorized Operator O(N r d)"
        metadata["key_mechanisms"] = ["Truncated SVD / Factorization", "Orthogonal Basis U and V^T", "Singular Spectrum Truncation", "Rank Bottleneck Projection"]
    elif any(k in full_text for k in ["quant", "int8", "int4", "fp4", "nf4", "smoothquant", "awq", "gptq", "quip", "hqq"]):
        metadata["family"] = "compression"
        metadata["accent_color"] = "#06b6d4" # Cyan
        metadata["complexity"]["time_complexity"] = "O(N d / \\gamma)"
        metadata["complexity"]["space_complexity"] = "O(N d \\cdot b)"
        metadata["complexity"]["scaling_exponent"] = 0.95
        metadata["complexity"]["description"] = "Non-linear / Sub-byte Quantization"
        metadata["key_mechanisms"] = ["Affine Scaling & Zero-Point", "Hessian-Weighted Error Min", "Asymmetric Codebook Mapping", "Outlier Channel Isolation"]
    else:
        metadata["family"] = "linear"
        metadata["accent_color"] = "#38bdf8"
        metadata["complexity"]["time_complexity"] = "O(N d^2)"
        metadata["complexity"]["space_complexity"] = "O(N d)"
        metadata["complexity"]["scaling_exponent"] = 1.0
        metadata["complexity"]["description"] = "Linear / Efficient Scaled Operator"
        metadata["key_mechanisms"] = ["Linear Projection", "Key-Value State Caching", "Associative Contraction", "Residual Normalization"]

    # Write metadata JSON
    json_path = os.path.join(METADATA_DIR, f"{mod_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return metadata

def run_extraction_all():
    att_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "Codes/framework/attentions/*.py")))
    comp_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "Codes/framework/compression_engine/*.py")))
    
    extracted = {}
    for f in att_files:
        if not f.endswith("__init__.py"):
            m = extract_metadata_from_file(f, "attention")
            if m: extracted[m["module_name"]] = m
            
    for f in comp_files:
        if not f.endswith("__init__.py"):
            m = extract_metadata_from_file(f, "compression")
            if m: extracted[m["module_name"]] = m
            
    print(f"Extracted metadata for {len(extracted)} modules into {METADATA_DIR}/")
    return extracted

if __name__ == "__main__":
    run_extraction_all()
