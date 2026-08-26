from pathlib import Path

p = Path("Codes/framework/run_quality_sweep.py")
content = p.read_text(encoding="utf-8")

# 1. Update evaluate_single_seed_run signature
old_sig = '''    baseline_logits_cache: Optional[Dict[Tuple[str, str, int], torch.Tensor]] = None,
) -> SeedResult:'''

new_sig = '''    baseline_logits_cache: Optional[Dict[Tuple[str, str, int], torch.Tensor]] = None,
    pretrained: bool = False,
    gradient_checkpointing: bool = False,
) -> SeedResult:'''

content = content.replace(old_sig, new_sig, 1)

# 2. Update model instantiation
old_inst = '''    # 1. Instantiate Model
    model, cfg = get_base_model_cached(
        model_family=model_family,
        num_labels=num_labels,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        device=device,
    )'''

new_inst = '''    # 1. Instantiate Model
    if pretrained:
        from transformers import AutoModelForSequenceClassification
        model_name_map = {
            "distilbert": "distilbert-base-uncased",
            "bert": "bert-base-uncased",
            "roberta": "roberta-base",
        }
        hf_model_id = model_name_map.get(model_family.lower(), "distilbert-base-uncased")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_model_id,
                num_labels=num_labels,
            )
            if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            cfg = model.config
        except Exception as e:
            logger.warning("Could not load pretrained weights for %s: %s. Falling back to base model.", hf_model_id, e)
            model, cfg = get_base_model_cached(
                model_family=model_family,
                num_labels=num_labels,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                device=device,
            )
    else:
        model, cfg = get_base_model_cached(
            model_family=model_family,
            num_labels=num_labels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            device=device,
        )'''

content = content.replace(old_inst, new_inst, 1)

# 3. Update run_quality_benchmark_sweep signature
old_sweep_sig = '''    output_dir: Optional[str] = None,
    allow_synthetic: bool = False,
) -> str:'''

new_sweep_sig = '''    output_dir: Optional[str] = None,
    allow_synthetic: bool = False,
    pretrained: bool = False,
    gradient_checkpointing: bool = False,
) -> str:'''

content = content.replace(old_sweep_sig, new_sweep_sig, 1)

# 4. Auto-scale dimensions if pretrained is true
old_cfg = '''    config_dict = {
        "run_id": run_id,'''

new_cfg = '''    if pretrained:
        if hidden_size == 128:
            if "distilbert" in active_models:
                hidden_size, num_heads, num_layers = 768, 12, 6
            else:
                hidden_size, num_heads, num_layers = 768, 12, 12
        if batch_size == 128:
            batch_size = 16

    config_dict = {
        "run_id": run_id,
        "pretrained": pretrained,
        "gradient_checkpointing": gradient_checkpointing,'''

content = content.replace(old_cfg, new_cfg, 1)

# 5. Pass pretrained and gradient_checkpointing to evaluate_single_seed_run call
old_call = '''                            allow_synthetic=allow_synthetic,
                            baseline_logits_cache=baseline_logits_cache,
                        )'''

new_call = '''                            allow_synthetic=allow_synthetic,
                            baseline_logits_cache=baseline_logits_cache,
                            pretrained=pretrained,
                            gradient_checkpointing=gradient_checkpointing,
                        )'''

content = content.replace(old_call, new_call, 1)

# 6. Add CLI args in main()
old_args = '''    parser.add_argument("--precision", type=str, default="fp16", help="Precision (fp16, bf16, fp32)")
    parser.add_argument("--output_dir", type=str, default=None, help="Target output directory")'''

new_args = '''    parser.add_argument("--precision", type=str, default="fp16", help="Precision (fp16, bf16, fp32)")
    parser.add_argument("--output_dir", type=str, default=None, help="Target output directory")
    parser.add_argument("--pretrained", action="store_true", default=False, help="Use real pretrained HuggingFace weights")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing")'''

content = content.replace(old_args, new_args, 1)

# 7. Pass to run_quality_benchmark_sweep in main()
old_main_call = '''        precision=args.precision,
        output_dir=args.output_dir,
    )'''

new_main_call = '''        precision=args.precision,
        output_dir=args.output_dir,
        pretrained=args.pretrained,
        gradient_checkpointing=args.gradient_checkpointing,
    )'''

content = content.replace(old_main_call, new_main_call, 1)

p.write_text(content, encoding="utf-8")
print("Codes/framework/run_quality_sweep.py patched successfully.")
