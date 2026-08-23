import re

with open("benchmark/run_benchmark.py", "r") as f:
    content = f.read()

# Add device selection to main()
device_code = """    output_dir = Path(config.get("output_dir", "benchmark_output"))

    # Setup Target Execution Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Target Device: CUDA ({torch.cuda.get_device_name()})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Target Device: Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Target Device: CPU")"""

content = content.replace('    output_dir = Path(config.get("output_dir", "benchmark_output"))', device_code)

# Fix dummy inputs
content = content.replace(
    'dummy_inputs = {\n                "input_ids": torch.randint(0, 1000, (4, 64)),\n                "attention_mask": torch.ones((4, 64)),\n            }',
    'dummy_inputs = {\n                "input_ids": torch.randint(0, 1000, (4, 64), device=device),\n                "attention_mask": torch.ones((4, 64), device=device),\n            }'
)

# Use regex to append .to(device) to model building steps
content = re.sub(
    r'(base_model, build_status = build_model_with_attention\([^)]+\))',
    r'\1\n        if base_model is not None:\n            base_model = base_model.to(device)',
    content
)

content = re.sub(
    r'(prof_model, _ = build_model_with_attention\([^)]+\))',
    r'\1\n            if prof_model is not None:\n                prof_model = prof_model.to(device)',
    content
)

content = re.sub(
    r'(diag_model, _ = build_model_with_attention\([^)]+\))',
    r'\1\n            if diag_model is not None:\n                diag_model = diag_model.to(device)',
    content
)

with open("benchmark/run_benchmark.py", "w") as f:
    f.write(content)

