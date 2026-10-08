"""
Auto-import all attentions modules across subpackages and export their registered classes.
"""

import importlib
import logging
import sys

from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["ATTENTION_REGISTRY"]

_SUBPACKAGES = ['adaptive', 'dense', 'experimental', 'fast_kernels', 'gated', 'hybrid', 'linear', 'modern', 'sparse', 'ssm', 'svd']

_LEGACY_MAP = {
    'lola_attention': 'adaptive.lola_attention', 'eat': 'adaptive.eat', 'progressive_token_pruning': 'adaptive.progressive_token_pruning',
    'infini_attention': 'adaptive.infini_attention', 'span_expanded_attention': 'adaptive.span_expanded_attention',
    'adaptive_span_attention': 'adaptive.adaptive_span_attention', 'squeezing_heads_distill': 'adaptive.squeezing_heads_distill',
    'token_compressed_attention': 'adaptive.token_compressed_attention', 'block_transformer_hiba': 'adaptive.block_transformer_hiba',
    'full_attention': 'dense.full_attention', 'zeros_attention': 'dense.zeros_attention',
    'template_attention': 'experimental.template_attention', 'xattention': 'fast_kernels.xattention',
    'flash_attention3_fp8': 'fast_kernels.flash_attention3_fp8', 'soa_attn': 'fast_kernels.soa_attn',
    'flash_attention3': 'fast_kernels.flash_attention3', 'adasplash_attention': 'fast_kernels.adasplash_attention',
    'gated_deltanet_block': 'gated.gated_deltanet_block', 'kda_attention': 'gated.kda_attention',
    'hgrn2_gated_outer': 'gated.hgrn2_gated_outer', 'hgrn2_gated_deltanet': 'gated.hgrn2_gated_deltanet',
    'hgrn2_gated_layer': 'gated.hgrn2_gated_layer', 'gated_delta_net': 'gated.gated_delta_net',
    'hla_attention': 'gated.hla_attention', 'decay_linear_attention': 'gated.decay_linear_attention',
    'gated_linear_attention': 'gated.gated_linear_attention', 'regla_attention': 'hybrid.regla_attention',
    'gated_hybrid_kernel_attention': 'hybrid.gated_hybrid_kernel_attention', 'spectron_fp8_attention': 'hybrid.spectron_fp8_attention',
    'optimized_hybrid_attention': 'hybrid.optimized_hybrid_attention', 'mixattention_moska': 'hybrid.mixattention_moska',
    'granite_hybrid': 'hybrid.granite_hybrid', 'landmark_local_attention': 'hybrid.landmark_local_attention',
    'refined_landmark_attention': 'hybrid.refined_landmark_attention', 'split_head_hybrid_attention': 'hybrid.split_head_hybrid_attention',
    'fusion_attn': 'hybrid.fusion_attn', 'tri_branch_hybrid_attention': 'hybrid.tri_branch_hybrid_attention',
    'positive_kernel_linear_attention': 'hybrid.positive_kernel_linear_attention', 'halo_regla': 'hybrid.halo_regla',
    'hybrid_local_kernel_attention': 'hybrid.hybrid_local_kernel_attention', 'hybrid_attention': 'hybrid.hybrid_attention',
    'linformer_attention': 'linear.linformer_attention', 'cosformer': 'linear.cosformer',
    'nystrom_attention': 'linear.nystrom_attention', 'nasc_attention': 'linear.nasc_attention',
    'tpa_attention': 'linear.tpa_attention', 'spectra_v2': 'linear.spectra_v2',
    'spectraformer_attention': 'linear.spectraformer_attention', 'spectron': 'linear.spectron',
    'performer_attention': 'linear.performer_attention', 'tlinformer_attention': 'linear.tlinformer_attention',
    'primal_attention': 'linear.primal_attention', 'rala_attention': 'linear.rala_attention',
    'rpc_attention': 'linear.rpc_attention', 'galerkin_attention': 'linear.galerkin_attention',
    'triton_cosformer': 'linear.triton_cosformer', 'chela_slay': 'linear.chela_slay',
    'unified_rope': 'modern.unified_rope', 'gqa_attention': 'modern.gqa_attention',
    'vig_attention': 'modern.vig_attention', 'mla_attention': 'modern.mla_attention',
    'wgqa_attention': 'modern.wgqa_attention', 'qcqa_attention': 'modern.qcqa_attention',
    'nha_attention': 'modern.nha_attention', 'padre_attention': 'sparse.padre_attention',
    'moba_attention': 'sparse.moba_attention', 'routing_attention': 'sparse.routing_attention',
    'moe_deltakv_paged': 'sparse.moe_deltakv_paged', 'cluster_attention': 'sparse.cluster_attention',
    'lsh_attention': 'sparse.lsh_attention', 'reap_moe': 'sparse.reap_moe',
    'nsa_attention': 'sparse.nsa_attention', 'bam_umoe': 'sparse.bam_umoe',
    'hamming_attention': 'sparse.hamming_attention', 'adaptive_sparse_attention': 'sparse.adaptive_sparse_attention',
    'expert_choice_moe': 'sparse.expert_choice_moe', 'vspec': 'ssm.vspec',
    'goldfinch_ahn': 'ssm.goldfinch_ahn', 'phantom': 'ssm.phantom',
    'mamba_block': 'ssm.mamba_block', 'selective_ssm': 'ssm.selective_ssm',
    'upi_stuffed_mamba': 'ssm.upi_stuffed_mamba', 'jamba': 'ssm.jamba',
    'nexus': 'ssm.nexus', 'hymba_transx': 'ssm.hymba_transx',
    'griffin_ssd': 'ssm.griffin_ssd', 'jamba_info': 'ssm.jamba_info',
    'samba_moe': 'ssm.samba_moe', 'mamba_svd_flash': 'ssm.mamba_svd_flash',
    'samba_m3': 'ssm.samba_m3', 'orion': 'ssm.orion',
    'spectral_mamba': 'ssm.spectral_mamba', 'simple_ssm': 'ssm.simple_ssm',
    'compressed_attention': 'svd.compressed_attention', 'svd_projection_attention': 'svd.svd_projection_attention',
    'fused_svd_kernel': 'svd.fused_svd_kernel', 'compressed_mha_attention': 'svd.compressed_mha_attention',
    'factorised_matrix_state': 'svd.factorised_matrix_state', 'projection_factorized_attn': 'svd.projection_factorized_attn',
    'svd_attention': 'svd.svd_attention', 'svd_compressed': 'svd.svd_compressed',
    'triton_svd_linear_attention': 'svd.triton_svd_linear_attention', 'asvd_attention': 'svd.asvd_attention',
    'naive_svd': 'svd.naive_svd', 'residual_svd_attention': 'svd.residual_svd_attention',
    'minimal_svd': 'svd.minimal_svd', 'lora_attention': 'svd.lora_attention',
    'svd_linear_decay': 'svd.svd_linear_decay', 'svd_llm_v2_attention': 'svd.svd_llm_v2_attention',
    'salt_attention': 'svd.salt_attention', 'svd_mamba': 'svd.svd_mamba',
    'minimal_nova': 'svd.minimal_nova', 'svd_linear_attention': 'svd.svd_linear_attention',
    'adaptive_svd_attention': 'svd.adaptive_svd_attention'
}

# Renamed module alias mappings: (subpath, {old_class_name: target_attr_name})
_RENAMED_MODULES = {
    'aegis_hybrid': ('hybrid.tri_branch_hybrid_attention', {'AegisHybridAttention': 'TriBranchHybridAttention', 'inject_aegis_hybrid': 'inject_tri_branch_hybrid'}),
    'aegis_spectron_fp8': ('hybrid.spectron_fp8_attention', {'AegisSpectronFP8Attention': 'SpectronFP8Attention', 'inject_aegis_spectron_fp8': 'inject_spectron_fp8'}),
    'aegis_v2': ('hybrid.hybrid_local_kernel_attention', {'AegisAttentionV2': 'HybridLocalKernelAttention', 'inject_aegis_v2': 'inject_hybrid_local_kernel'}),
    'aegis_v5': ('hybrid.split_head_hybrid_attention', {'AegisAttention': 'SplitHeadHybridAttention', 'AegisConfigV5': 'SplitHeadHybridConfig', 'inject_aegis_v5': 'inject_split_head_hybrid'}),
    'aegis_v6': ('hybrid.gated_hybrid_kernel_attention', {'AegisAttentionV3': 'GatedHybridKernelAttention', 'inject_aegis_v6': 'inject_gated_hybrid_kernel'}),
    'leka_v1': ('hybrid.positive_kernel_linear_attention', {'LEKAV1Attention': 'PositiveKernelLinearAttention', 'inject_leka_v1': 'inject_positive_kernel_linear'}),
    'leka_v2': ('hybrid.landmark_local_attention', {'LEKAv2Attention': 'LandmarkLocalAttention', 'inject_leka_v2': 'inject_landmark_local'}),
    'leka_v3': ('hybrid.refined_landmark_attention', {'LEKAv3Attention': 'RefinedLandmarkAttention', 'inject_leka_v3': 'inject_refined_landmark'}),
    'res_svd': ('svd.residual_svd_attention', {'ResSVDAttention': 'ResidualSVDAttention', 'inject_res_svd': 'inject_residual_svd_attention'}),
}

# 1. Backward compatibility shim for sys.modules and import hooks (old flat import paths -> new subpackages)
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader
import types

def _get_target_module(leaf):
    if leaf in _RENAMED_MODULES:
        rel_path, class_map = _RENAMED_MODULES[leaf]
        real_mod = importlib.import_module(f".{rel_path}", __package__)
        alias_mod = types.ModuleType(f"{__name__}.{leaf}")
        alias_mod.__dict__.update(real_mod.__dict__)
        for old_k, target_k in class_map.items():
            if hasattr(real_mod, target_k):
                setattr(alias_mod, old_k, getattr(real_mod, target_k))
        return alias_mod
    elif leaf in _LEGACY_MAP:
        return importlib.import_module(f".{_LEGACY_MAP[leaf]}", __package__)
    elif leaf == 'replace':
        return importlib.import_module("neuralops.core.replace_attention")
    return None

class _LegacyAttnLoader(Loader):
    def __init__(self, leaf):
        self.leaf = leaf
    def create_module(self, spec):
        return _get_target_module(self.leaf)
    def exec_module(self, module):
        pass

class _LegacyAttnFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(__name__ + "."):
            leaf = fullname[len(__name__) + 1:]
            if leaf in _LEGACY_MAP or leaf in _RENAMED_MODULES or leaf == 'replace':
                return spec_from_loader(fullname, _LegacyAttnLoader(leaf))
        return None

sys.meta_path.insert(0, _LegacyAttnFinder())

# Pre-populate sys.modules aliases (keep 'replace' dynamic on import so it doesn't affect canonical registry keys)
for _old in list(_LEGACY_MAP.keys()) + list(_RENAMED_MODULES.keys()):
    try:
        _mod = _get_target_module(_old)
        if _mod is not None:
            sys.modules[f"{__name__}.{_old}"] = _mod
    except Exception as _e:
        logger.debug(f"Initial legacy mapping deferred for {_old}: {_e}")

def __getattr__(name):
    if name in _LEGACY_MAP or name in _RENAMED_MODULES or name == 'replace':
        mod = _get_target_module(name)
        if mod is not None:
            globals()[name] = mod
            return mod
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 2. Import all subpackages and re-export classes at top level
for _sub in _SUBPACKAGES:
    try:
        _submod = importlib.import_module(f".{_sub}", __package__)
        for _attr in dir(_submod):
            if not _attr.startswith("_"):
                _val = getattr(_submod, _attr)
                if isinstance(_val, type):
                    globals()[_attr] = _val
                    if _attr not in __all__:
                        __all__.append(_attr)
    except Exception as _e:
        logger.debug(f"Failed importing subpackage {_sub}: {_e}")
