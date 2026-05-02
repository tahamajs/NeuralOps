# framework/config_loader.py
"""Load YAML configuration and merge with CLI arguments."""

import yaml
import os
from framework.config import CompressionConfig

def load_yaml_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_compression_config(yaml_dict: dict, args=None) -> CompressionConfig:
    """
    Build a CompressionConfig from YAML sections, optionally overriding with CLI args.
    """
    cfg = CompressionConfig()

    # Helper to set attributes from a dict
    def _apply_section(section, target_dict):
        for key, value in section.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    # Apply from YAML sections
    if 'attention_config' in yaml_dict:
        _apply_section(yaml_dict['attention_config'], {})
    if 'training' in yaml_dict:
        _apply_section(yaml_dict['training'], {})
    if 'lora' in yaml_dict:
        _apply_section(yaml_dict['lora'], {})
    if 'dora' in yaml_dict:
        _apply_section(yaml_dict['dora'], {})
    if 'rdrop' in yaml_dict:
        _apply_section(yaml_dict['rdrop'], {})
    if 'r3f' in yaml_dict:
        _apply_section(yaml_dict['r3f'], {})
    if 'mixout' in yaml_dict:
        _apply_section(yaml_dict['mixout'], {})
    if 'llrd' in yaml_dict:
        _apply_section(yaml_dict['llrd'], {})
    if 'dislora' in yaml_dict:
        _apply_section(yaml_dict['dislora'], {})
    if 'fisher' in yaml_dict:
        _apply_section(yaml_dict['fisher'], {})
    if 'gelora' in yaml_dict:
        _apply_section(yaml_dict['gelora'], {})
    if 'hardware' in yaml_dict:
        _apply_section(yaml_dict['hardware'], {})
    if 'memory' in yaml_dict:
        _apply_section(yaml_dict['memory'], {})
    if 'model' in yaml_dict:
        _apply_section(yaml_dict['model'], {})
        if 'attention_type' in yaml_dict['model']:
            cfg.attn_type = yaml_dict['model']['attention_type']

    # Override with CLI arguments (if provided)
    if args:
        for key in ['batch_size', 'epochs', 'lr', 'max_len', 'attn_type',
                    'lora_r', 'lora_alpha', 'lora_dropout', 'rdrop_alpha',
                    'r3f_epsilon', 'mixout_p', 'llrd_decay', 'dislora_beta',
                    'dislora_capacity_factor', 'gelora_margin', 'gelora_min_rank',
                    'gelora_max_rank', 'fisher_scale_max', 'num_landmarks',
                    'feature_dim', 'k_early', 'k_mid', 'k_late', 'use_gate',
                    'block_size', 'rff_features', 'cur_rank', 'lsh_rounds',
                    'lsh_bucket_size', 'svd_energy', 'svd_rank_qkv',
                    'svd_rank_out', 'compression_rank', 'softpick_enabled',
                    'armor_enabled', 'ast_enabled', 'deltakv_enabled',
                    'kvt_enabled']:
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(cfg, key, getattr(args, key))

    return cfg