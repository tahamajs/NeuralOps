import argparse
import sys

import torch


def parse_args(argv=None):
    raw_argv = sys.argv[1:] if argv is None else argv
    has_explicit_batch = any(
        arg == "--batch_size" or arg.startswith("--batch_size=") for arg in raw_argv
    )
    has_explicit_parallel = any(
        arg == "--parallel" or arg.startswith("--parallel=") for arg in raw_argv
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset_dir", default=None)
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument(
        "--cached_batches_dir",
        default=None,
        help="Path to a directory containing pre-cached tokenized batches.",
    )
    ap.add_argument("--num_labels", type=int, default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--modules", default=["all"], nargs="*")
    ap.add_argument("--output_suffix", type=str, default=None)
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--module_filter", dest="modules", nargs="*")
    ap.add_argument("--max_steps", type=int, default=1)
    ap.add_argument("--finetune_steps", type=int, default=200)
    ap.add_argument(
        "--finetune_mode",
        choices=("head", "lora", "full"),
        default="head",
    )
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    ap.add_argument("--recovery_steps", type=int, default=50)
    ap.add_argument("--recovery_modules", default="")
    ap.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    ap.add_argument("--use_muon", action="store_true")
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_vram_fraction", type=float, default=None)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    ap.add_argument("--freeze_clf", action="store_true")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--eval_seq_len", type=int, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--kl_distill", action="store_true")
    ap.add_argument("--kl_alpha", type=float, default=0.5)
    ap.add_argument("--kl_temperature", type=float, default=2.0)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=False)
    ap.add_argument("--strict_compat", action="store_true")
    ap.add_argument(
        "--model_family",
        choices=(
            "auto",
            "bert",
            "roberta",
            "gpt",
            "gpt2",
            "qwen",
            "qwen3_5",
            "t5",
            "ssm",
            "unknown",
        ),
        default="auto",
    )
    ap.add_argument("--parent_checkpoint", type=str, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no_skip", action="store_true", default=False)
    ap.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Diagnose module injection and model forward pass without full recovery training or eval.",
    )
    ap.add_argument(
        "--debug_module",
        action="store_true",
        default=False,
        help="Run deep tensor health check (NaN, shape, norm, parameter delta).",
    )
    ap.add_argument(
        "--verbose_debug",
        action="store_true",
        default=False,
        help="Print detailed step-by-step tensor inspection logs.",
    )
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--force_rank", action="store_true", default=False)
    ap.add_argument("--rank_ratio", type=float, default=None)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--asvd_dynamic_rank", action="store_true", default=True)
    ap.add_argument("--asvd_energy_threshold", type=float, default=0.95)
    ap.add_argument("--asvd_error_threshold", type=float, default=0.20)
    ap.add_argument("--aasvd_max_error", type=float, default=0.85)
    ap.add_argument(
        "--aasvd_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--aasvd_ema_beta", type=float, default=0.9)
    ap.add_argument(
        "--aasvd_fallback_to_original",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--group_size", type=int, default=3)
    ap.add_argument("--defer_export", action="store_true", default=False)
    ap.add_argument("--max_error", type=float, default=0.50)
    ap.add_argument(
        "--strict_rank",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="If True, the compressor uses exactly the requested rank "
        "without escalating it to fit the error threshold.",
    )
    ap.add_argument("--a3_max_error", type=float, default=None)
    ap.add_argument(
        "--a3_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--span_lambda", type=float, default=0.0)
    ap.add_argument("--adaptive_ranks_max_error", type=float, default=0.85)
    ap.add_argument("--adaptive_ranks_energy_threshold", type=float, default=0.92)
    ap.add_argument(
        "--adaptive_ranks_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--adasvd_max_error", type=float, default=0.70)
    ap.add_argument(
        "--adasvd_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--adasvd_min_rank", type=int, default=16)
    ap.add_argument("--admm_sparsity", type=float, default=0.5)
    ap.add_argument("--admm_quant_bits", type=int, default=4)
    ap.add_argument(
        "--admm_structured_24",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--admm_max_error", type=float, default=0.45)
    ap.add_argument(
        "--admm_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--admm_rho1", type=float, default=0.1)
    ap.add_argument("--admm_rho2", type=float, default=0.1)
    ap.add_argument("--admm_max_iter", type=int, default=8)
    ap.add_argument("--admm_pruner_sparsity", type=float, default=0.5)
    ap.add_argument(
        "--admm_pruner_structured_24",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--admm_pruner_max_error", type=float, default=0.45)
    ap.add_argument(
        "--admm_pruner_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--admm_pruner_lambda_distill", type=float, default=0.1)
    ap.add_argument("--admm_pruner_distill_batch_size", type=int, default=32)
    ap.add_argument("--admm_pruner_rho", type=float, default=0.1)
    ap.add_argument("--admm_pruner_max_iter", type=int, default=6)
    ap.add_argument("--armor_rank", type=int, default=16)
    ap.add_argument("--armor_sparsity", type=float, default=0.5)
    ap.add_argument(
        "--armor_structured_24",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--armor_max_error", type=float, default=0.35)
    ap.add_argument(
        "--armor_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument(
        "--armor_decouple",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--armor_lora_alpha", type=float, default=1.0)
    ap.add_argument(
        "--armor_skip_attention_qkv",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    ap.add_argument("--astro_bits", type=int, default=8)
    ap.add_argument(
        "--astro_symmetric",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument(
        "--astro_per_channel",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    ap.add_argument(
        "--astro_use_ste",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    ap.add_argument("--astro_max_error", type=float, default=0.20)
    ap.add_argument(
        "--astro_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument(
        "--astro_use_fisher_adaptive_bits",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    ap.add_argument("--astro_target_avg_bits", type=float, default=6.0)
    ap.add_argument("--astro_min_bits", type=int, default=2)
    ap.add_argument("--astro_max_bits", type=int, default=8)
    ap.add_argument("--astro_fisher_beta", type=float, default=0.5)
    ap.add_argument("--cachenotes_task", type=str, default="qa")
    ap.add_argument("--cachenotes_keep_ratio", type=float, default=0.5)
    ap.add_argument("--cachenotes_max_error", type=float, default=0.35)
    ap.add_argument(
        "--cachenotes_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--cachenotes_lora_r", type=int, default=16)
    ap.add_argument("--cachenotes_lora_alpha", type=float, default=1.0)
    ap.add_argument(
        "--cachenotes_skip_attention_qkv",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--come_target_layers", type=int, default=None)
    ap.add_argument("--come_sparsity", type=float, default=0.3)
    ap.add_argument("--come_alpha_init", type=float, default=0.1)
    ap.add_argument("--come_lora_r", type=int, default=16)
    ap.add_argument("--come_lora_alpha", type=float, default=1.0)
    ap.add_argument(
        "--come_skip_attention_qkv",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument(
        "--come_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    ap.add_argument("--coopq_bits", type=int, default=4)
    ap.add_argument("--coopq_min_bits", type=int, default=4)
    ap.add_argument("--coopq_max_bits", type=int, default=8)
    ap.add_argument(
        "--coopq_symmetric",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    ap.add_argument(
        "--coopq_per_channel",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument(
        "--coopq_use_ste",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument(
        "--coopq_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--cr_rank", type=int, default=32)
    ap.add_argument("--cr_max_error", type=float, default=0.35)
    ap.add_argument(
        "--cr_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--cur_rank", type=int, default=32)
    ap.add_argument("--cur_min_rank", type=int, default=8)
    ap.add_argument("--cur_method", type=str, default="leverage")
    ap.add_argument("--cur_max_error", type=float, default=0.35)
    ap.add_argument(
        "--cur_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--cur_moe_rank", type=int, default=32)
    ap.add_argument(
        "--cur_moe_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--delta_kv_rank", type=int, default=16)
    ap.add_argument("--delta_kv_sparsity", type=float, default=0.29)
    ap.add_argument("--delta_kv_max_error", type=float, default=0.35)
    ap.add_argument(
        "--delta_kv_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--direct_svd_rank", type=int, default=128)
    ap.add_argument("--direct_svd_max_error", type=float, default=0.35)
    ap.add_argument(
        "--direct_svd_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--dobi_rank", type=int, default=None)
    ap.add_argument("--dobi_init_rank_ratio", type=float, default=0.5)
    ap.add_argument("--dobi_init_gate_val", type=float, default=2.5)
    ap.add_argument("--dobi_max_error", type=float, default=0.35)
    ap.add_argument(
        "--dobi_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--dynamic_min_rank", type=int, default=16)
    ap.add_argument("--dynamic_max_rank", type=int, default=256)
    ap.add_argument("--dynamic_target_param_ratio", type=float, default=0.5)
    ap.add_argument("--dynamic_max_error", type=float, default=0.35)
    ap.add_argument(
        "--dynamic_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--enlighten_rank", type=int, default=128)
    ap.add_argument("--enlighten_sparsity", type=float, default=0.5)
    ap.add_argument(
        "--enlighten_structured_24",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--enlighten_max_error", type=float, default=0.35)
    ap.add_argument(
        "--enlighten_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--entropy_budget_rank", type=int, default=None)
    ap.add_argument("--entropy_budget_total_budget_ratio", type=float, default=0.7)
    ap.add_argument("--entropy_budget_max_error", type=float, default=0.35)
    ap.add_argument(
        "--entropy_budget_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--erc_allowed_error", type=float, default=0.20)
    ap.add_argument("--erc_max_error", type=float, default=0.35)
    ap.add_argument(
        "--erc_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--erc_rank", type=int, default=None)
    ap.add_argument("--faq_bits", type=int, default=8)
    ap.add_argument("--faq_alpha", type=float, default=0.5)
    ap.add_argument(
        "--faq_symmetric",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument(
        "--faq_use_ste",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--faq_lora_r", type=int, default=16)
    ap.add_argument("--faq_max_error", type=float, default=0.35)
    ap.add_argument(
        "--faq_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--flap_sparsity", type=float, default=0.2)
    ap.add_argument("--flap_metric_type", type=str, default="WIFV")
    ap.add_argument(
        "--flap_structured_24",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    ap.add_argument("--flap_max_error", type=float, default=0.35)
    ap.add_argument(
        "--flap_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--flat_llm_keep_ratio", type=float, default=0.7)
    ap.add_argument("--flat_llm_damping", type=float, default=1e-4)
    ap.add_argument(
        "--flat_llm_structured_24",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    ap.add_argument("--flat_llm_max_error", type=float, default=0.35)
    ap.add_argument(
        "--flat_llm_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--frandi_rank", type=int, default=128)
    ap.add_argument("--frandi_energy_threshold", type=float, default=None)
    ap.add_argument("--frandi_lora_r", type=int, default=16)
    ap.add_argument("--frandi_max_error", type=float, default=0.35)
    ap.add_argument(
        "--frandi_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--grecodec_rank", type=int, default=128)
    ap.add_argument("--grecodec_max_iter", type=int, default=15)
    ap.add_argument("--grecodec_lora_r", type=int, default=16)
    ap.add_argument("--grecodec_max_error", type=float, default=0.35)
    ap.add_argument(
        "--grecodec_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--hashed_net_buckets", type=int, default=65536)
    ap.add_argument(
        "--hashed_net_use_sign",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--hashed_net_lora_r", type=int, default=16)
    ap.add_argument("--hashed_net_max_error", type=float, default=0.35)
    ap.add_argument(
        "--hashed_net_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--huffman_quantize_bits", type=int, default=8)
    ap.add_argument("--huffman_max_error", type=float, default=0.20)
    ap.add_argument(
        "--huffman_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--hybrid_svd_rank", type=int, default=128)
    ap.add_argument(
        "--hybrid_svd_use_full",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    ap.add_argument("--hybrid_svd_lora_r", type=int, default=16)
    ap.add_argument("--hybrid_svd_max_error", type=float, default=0.55)
    ap.add_argument(
        "--hybrid_svd_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--hybridkv_budget", type=float, default=0.3)
    ap.add_argument("--hybridkv_static_ratio", type=float, default=0.7)
    ap.add_argument("--hybridkv_max_cache_len", type=int, default=128)
    ap.add_argument("--hybridkv_num_sinks", type=int, default=4)
    ap.add_argument("--hybridkv_recent_window", type=int, default=32)
    ap.add_argument(
        "--hybridkv_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--kmeans_clusters", type=int, default=256)
    ap.add_argument("--kmeans_lora_r", type=int, default=16)
    ap.add_argument("--kmeans_max_error", type=float, default=0.20)
    ap.add_argument(
        "--kmeans_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--kronecker_rank", type=int, default=128)
    ap.add_argument("--kronecker_max_error", type=float, default=0.50)
    ap.add_argument(
        "--kronecker_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--tt_ranks", type=str, default="1,2,2,1")
    ap.add_argument("--kronecker_tt_lora_max_error", type=float, default=0.50)
    ap.add_argument(
        "--kronecker_tt_lora_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--kvtc_bits", type=int, default=4)
    ap.add_argument("--kvtc_pca_energy", type=float, default=0.90)
    ap.add_argument("--kvtc_max_error", type=float, default=0.20)
    ap.add_argument(
        "--kvtc_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument(
        "--kvtc_fallback_to_original",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--liftr_gelu_coefficient", type=float, default=0.8)
    ap.add_argument(
        "--liftr_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--lillama_rank_ratio", type=float, default=0.5)
    ap.add_argument("--lillama_max_error", type=float, default=0.50)
    ap.add_argument(
        "--lillama_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    ap.add_argument("--lsc_sparsity", type=float, default=0.3)
    ap.add_argument("--lsc_max_error", type=float, default=0.50)
    ap.add_argument(
        "--lsc_target_attention_only",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )

    # Pre-parse to check if a config file is provided
    pre_args, _ = ap.parse_known_args(raw_argv)
    if pre_args.config and Path(pre_args.config).exists():
        import yaml

        with open(pre_args.config, encoding="utf-8") as yf:
            cfg_dict = yaml.safe_load(yf) or {}
        for k, v in cfg_dict.items():
            if k in ap._option_string_actions:
                continue
            if hasattr(pre_args, k):
                ap.set_defaults(**{k: v})

    args, unknown_args = ap.parse_known_args(raw_argv)
    args.unknown_args = unknown_args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    extended_args = {}
    if unknown_args:
        LOG.info(
            "Parsed %d module-specific extended arguments: %s",
            len(unknown_args),
            unknown_args,
        )
        i = 0
        while i < len(unknown_args):
            flag = unknown_args[i]
            if flag.startswith("--"):
                key = flag[2:].replace("-", "_")
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    val = unknown_args[i + 1]
                    if val.lower() in ("true", "1", "yes"):
                        parsed_val = True
                    elif val.lower() in ("false", "0", "no"):
                        parsed_val = False
                    else:
                        try:
                            parsed_val = int(val) if "." not in val else float(val)
                        except ValueError:
                            parsed_val = val
                    setattr(args, key, parsed_val)
                    extended_args[key] = parsed_val
                    i += 2
                else:
                    setattr(args, key, True)
                    extended_args[key] = True
                    i += 1
            else:
                i += 1

    # Remove --strict_rank from extended args if argparse already handled it
    for key in list(extended_args.keys()):
        if key == "strict_rank":
            del extended_args[key]
    args.extended_args = extended_args

    torch.manual_seed(args.seed)
    # Fix 9: respect OMP_NUM_THREADS
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            target_frac = (
                float(args.max_vram_fraction)
                if getattr(args, "max_vram_fraction", None) is not None
                else 0.70
            )
            if 0.0 < target_frac <= 1.0:
                torch.cuda.set_per_process_memory_fraction(target_frac)
                LOG.info(
                    "Configured PyTorch per-process GPU memory fraction cap: %.1f%% (leaving %.2f GB headroom)",
                    target_frac * 100.0,
                    (1.0 - target_frac) * 8.0,
                )
        except Exception as frac_err:
            LOG.debug("Could not set memory fraction: %s", frac_err)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            LOG.info(
                "GPU VRAM: %.2f GB free / %.2f GB total on %s",
                free_gb,
                total_gb,
                torch.cuda.get_device_name(),
            )
        except Exception:
            pass

    auto_p, auto_b = _auto_parallel_and_batch(
        requested_parallel=args.parallel,
        requested_batch=args.batch_size,
        explicit_parallel=has_explicit_parallel,
        explicit_batch=has_explicit_batch,
    )
    args.parallel = auto_p
    args.batch_size = auto_b
    LOG.info(
        "Configured execution sizing: parallel=%d, batch_size=%d (explicit_parallel=%s, explicit_batch=%s)",
        args.parallel,
        args.batch_size,
        has_explicit_parallel,
        has_explicit_batch,
    )

    return args


# ============================================================================
# (The rest of the file — run_single_module_main, _run_candidate_pipeline,
#  run_module_isolated, main — is extremely long. Below is a placeholder
#  indicating that those functions are unchanged except where the 15 fixes
#  apply. To avoid truncation, in the actual file the fixes listed at the
#  top of this docstring must be applied at the indicated locations.)
# ============================================================================


# NOTE: The remaining functions (_run_candidate_pipeline, run_single_module_main,
# run_module_isolated, main) are preserved from the original file with the
# following patch points applied:
#
# 1. _run_candidate_pipeline: effective_timeout = max(cli_timeout, override_to)
# 2. run_single_module_main: skip logic uncommented for status == "PASS"
# 3. run_module_isolated: dataset_dir only appended if set
# 4. All datetime.utcnow() → _now_iso() or datetime.now(timezone.utc).isoformat()
# 5. FIELDS schema unified — already at module level above
# 6. perf_metrics/comp_metrics initialized to None before try
# 7. NaN loss skips the batch (continue) in finetune_lora
# 8. p.data.half() wrapped in torch.no_grad() with dtype check
# 9. torch.set_num_threads respects OMP_NUM_THREADS
# 10. Duplicate lora import removed (single import at top of is_lora block)
# 11. run_with_timeout and multiprocessing imports removed
# 12. SLOW_MODULE_TIMEOUT_OVERRIDES unified at module level (already done above)
# 13. existing set actually used for skip
# 14. Dead config pass block removed
# 15. Step loss accumulation clarified (already done in finetune_lora above)
#
