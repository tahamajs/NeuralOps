# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Internal module alias finder for deprecated legacy import paths.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any
import warnings

from importlib.abc import MetaPathFinder

_logger = logging.getLogger("neuralops.core._compat_importer")


class _AliasLoader:
    def __init__(self, mod: Any) -> None:
        self.mod = mod

    def create_module(self, _spec: Any) -> Any:
        return self.mod

    def exec_module(self, module: Any) -> None:
        pass


class _ModuleAliasFinder(MetaPathFinder):
    _ALIASES = {
        "neuralops.attentions.gqa": "neuralops.attentions.gqa_attention",
        "neuralops.attentions.galerkin": "neuralops.attentions.galerkin_attention",
        "neuralops.attentions.galerkin_inject": "neuralops.attentions.galerkin_attention",
        "neuralops.attentions.adaptive_span_inject": "neuralops.attentions.adaptive_span_attention",
        "neuralops.attentions.adaptive_sparse_inject": "neuralops.attentions.adaptive_sparse_attention",
        "neuralops.attentions.adasplash_inject": "neuralops.attentions.adasplash_attention",
        "neuralops.attentions.aegis_hybrid_attention": "neuralops.attentions.aegis_hybrid",
        "neuralops.attentions.aegis_v1": "neuralops.attentions.aegis_v2",
        "neuralops.attentions.aegis_v3": "neuralops.attentions.aegis_v2",
        "neuralops.attentions.aegis_v4": "neuralops.attentions.aegis_v2",
        "neuralops.attentions.minimal_aurora": "neuralops.attentions.minimal_svd",
        "neuralops.attentions.minimal_prism": "neuralops.attentions.minimal_svd",
        "neuralops.attentions.minimal_svd_v2": "neuralops.attentions.minimal_svd",
        "neuralops.attentions.cluster_inject": "neuralops.attentions.cluster_attention",
        "neuralops.attentions.compressed_mha": "neuralops.attentions.compressed_mha_attention",
        "neuralops.attentions.cosformer_inject": "neuralops.attentions.cosformer",
        "neuralops.attentions.hla_inject": "neuralops.attentions.hla_attention",
        "neuralops.attentions.kda_inject": "neuralops.attentions.kda_attention",
        "neuralops.attentions.leka_v1_inject": "neuralops.attentions.leka_v1",
        "neuralops.attentions.leka_v2_inject": "neuralops.attentions.leka_v2",
        "neuralops.attentions.leka_v3_inject": "neuralops.attentions.leka_v3",
        "neuralops.attentions.Mamba": "neuralops.attentions.mamba_svd_flash",
        "neuralops.attentions.SVD-Mamba": "neuralops.attentions.svd_mamba",
        "neuralops.attentions.mamba_attention": "neuralops.attentions.mamba_block",
        "neuralops.attentions.moba": "neuralops.attentions.moba_attention",
        "neuralops.attentions.nha_compressor": "neuralops.attentions.nha_attention",
        "neuralops.attentions.rpc": "neuralops.attentions.rpc_attention",
        "neuralops.attentions.salt": "neuralops.attentions.salt_attention",
        "neuralops.attentions.soa_fused_module": "neuralops.attentions.soa_attn",
        "neuralops.attentions.SOAFusedAttention": "neuralops.attentions.soa_attn",
        "neuralops.attentions.soa_module": "neuralops.attentions.soa_attn",
        "neuralops.attentions.spectra_v2_attention": "neuralops.attentions.spectra_v2",
        "neuralops.attentions.svd_compressed_attention": "neuralops.attentions.svd_compressed",
        "neuralops.attentions.svd_llm_v2": "neuralops.attentions.svd_llm_v2_attention",
        "neuralops.attentions.svd_proj_attn": "neuralops.attentions.svd_projection_attention",
        "neuralops.attentions.token_compressed": "neuralops.attentions.token_compressed_attention",
        "neuralops.attentions.tt": "neuralops.attentions.token_compressed_attention",
        "neuralops.attentions.upi_mamba": "neuralops.attentions.upi_stuffed_mamba",
        "neuralops.attentions.mamba": "neuralops.attentions.mamba_block",
        "neuralops.attentions.mamba_inject": "neuralops.attentions.mamba_block",
        "neuralops.attentions.simple_ssm": "neuralops.attentions.selective_ssm",
        "neuralops.attentions.smart_weighted_ensemble": "neuralops.attentions.hybrid_attention",
        "neuralops.attentions.weighted_ensemble": "neuralops.attentions.hybrid_attention",
        "neuralops.attentions.spectron_attention": "neuralops.attentions.spectron",
        "neuralops.attentions.squeezing_heads": "neuralops.attentions.squeezing_heads_distill",
        "neuralops.attentions.squeezing_heads_distiller": "neuralops.attentions.squeezing_heads_distill",
        "neuralops.attentions.spectral_mamba_attention": "neuralops.attentions.spectral_mamba",
        "neuralops.attentions.svd_sequence_attention": "neuralops.attentions.svd_attention",
        "neuralops.attentions.rala_compressor": "neuralops.attentions.rala_attention",
        "neuralops.attentions.selective_scan": "neuralops.core.selective_scan",
        "neuralops.attentions.ssm_scan": "neuralops.core.selective_scan",
        "neuralops.attentions.legacy_ensemble": "neuralops.core.legacy_ensemble",
        "neuralops.compression_engine.aasvd": "neuralops.compression_engine.aa_svd_compressor",
        "neuralops.compression_engine.adasvd": "neuralops.compression_engine.adasvd_compressor",
        "neuralops.compression_engine.cur_moe": "neuralops.compression_engine.cur_moe_compressor",
        "neuralops.compression_engine.frandi": "neuralops.compression_engine.frandi_compressor",
        "neuralops.compression_engine.grecodec": "neuralops.compression_engine.grecodec_compressor",
        "neuralops.compression_engine.hybridkv": "neuralops.compression_engine.hybridkv_compressor",
        "neuralops.compression_engine.kvtc": "neuralops.compression_engine.kvtc_compressor",
        "neuralops.compression_engine.modegpt": "neuralops.compression_engine.modegpt_compressor",
        "neuralops.compression_engine.oats": "neuralops.compression_engine.oats_compressor",
        "neuralops.compression_engine.proxsparse_pruner": "neuralops.compression_engine.proxsparse",
        "neuralops.compression_engine.rpca": "neuralops.compression_engine.rpca_compressor",
        "neuralops.compression_engine.saes_svd": "neuralops.compression_engine.saes_svd_v2",
        "neuralops.compression_engine.saes_svd_compressor": "neuralops.compression_engine.saes_svd_v2",
        "neuralops.compression_engine.sdrm": "neuralops.compression_engine.sdrm_compressor",
        "neuralops.compression_engine.slicegpt_compressor": "neuralops.compression_engine.slicegpt",
        "neuralops.compression_engine.soka": "neuralops.compression_engine.soka_compressor",
        "neuralops.compression_engine.swift_svd_compressor": "neuralops.compression_engine.swift_svd_act",
        "neuralops.compression_engine.zerokv_neo": "neuralops.compression_engine.zerokv_neo_compressor",
        "neuralops.compression_engine.cure": "neuralops.compression_engine.cur_moe_compressor",
        "neuralops.compression_engine.dobi": "neuralops.compression_engine.dobi_svd",
        "neuralops.compression_engine.dynamic_entropy": "neuralops.compression_engine.dynamic_rank",
        "neuralops.compression_engine.tomoe": "neuralops.compression_engine.tomoe_compressor",
        "neuralops.compression_engine.adaptive_compressor": "neuralops.compression_engine.adaptive_ranks",
        "neuralops.compression_engine.padre_attention": "neuralops.attentions.padre_attention",
        "neuralops.compression_engine.spectraformer_attention": "neuralops.attentions.spectraformer_attention",
        "neuralops.compression_engine.tlinformer_attention": "neuralops.attentions.tlinformer_attention",
        "neuralops.compression_engine.pruning_utils": "neuralops.core.pruning_utils",
        "neuralops.compression_engine.cur_linear": "neuralops.compression_engine.cur_compressor",
        "neuralops.compression_engine.kronecker_linear": "neuralops.compression_engine.kronecker_compressor",
        "neuralops.compression_engine.tt_linear": "neuralops.compression_engine.tt_compressor",
        "neuralops.compression_engine.hashed_linear": "neuralops.compression_engine.hashed_net_compressor",
        "neuralops.compression_engine.come": "neuralops.compression_engine.come_compressor",
        "neuralops.compression_engine.rala_compressor": "neuralops.attentions.rala_attention",
        "neuralops.run_quality_sweep": "neuralops.benchmarks.run_quality_sweep",
        "neuralops.run_rank_sweeper": "neuralops.benchmarks.run_rank_sweeper",
        "neuralops.run_all_benchmarks": "neuralops.benchmarks.run_all_benchmarks",
        "neuralops.run_gpt2_sweep": "neuralops.benchmarks.run_gpt2_sweep",
        "neuralops.run_long_context_sweep": "neuralops.benchmarks.run_long_context_sweep",
        "neuralops.run_seqlen_scaling": "neuralops.benchmarks.run_seqlen_scaling",
        "neuralops.run_seqlen_scaling_bench": "neuralops.benchmarks.run_seqlen_scaling_bench",
        "neuralops.run_multiseed": "neuralops.benchmarks.run_multiseed",
        "neuralops.run_multiseed_rigor": "neuralops.benchmarks.run_multiseed_rigor",
        "neuralops.run_one_module_isolated": "neuralops.benchmarks.run_one_module_isolated",
        "neuralops.run_ruler_benchmark": "neuralops.benchmarks.run_ruler_benchmark",
        "neuralops.run_whitened_svd": "neuralops.benchmarks.run_whitened_svd",
        "neuralops.launch_parallel_sweep": "neuralops.benchmarks.launch_parallel_sweep",
        "neuralops.run_all_modules_scheduler": "neuralops.benchmarks.run_all_modules_scheduler",
        "neuralops.kd_quick_test": "neuralops.benchmarks.kd_quick_test",
        "neuralops.eval_all_models_datasets": "neuralops.evaluation.eval_all_models_datasets",
        "neuralops.eval_perplexity": "neuralops.evaluation.eval_perplexity",
        "neuralops.full_mnli_validation": "neuralops.evaluation.full_mnli_validation",
        "neuralops.run_lm_eval_harness": "neuralops.evaluation.run_lm_eval_harness",
        "neuralops.run_baseline_comparison": "neuralops.evaluation.run_baseline_comparison",
        "neuralops.train_lra": "neuralops.trainer.train_lra",
        "neuralops.generate_all_modules_analysis": "neuralops.analysis.generate_all_modules_analysis",
        "neuralops.core.triton_kernels.block_sparse_streaming_attn": "neuralops.core.triton_kernels.standalone.block_sparse_streaming_attn",
        "neuralops.core.triton_kernels.chunked_delta_gate_scan": "neuralops.core.triton_kernels.standalone.chunked_delta_gate_scan",
        "neuralops.core.triton_kernels.cosformer_linear_attention": "neuralops.core.triton_kernels.standalone.cosformer_linear_attention",
        "neuralops.core.triton_kernels.fused_act_scale_quant_gemm": "neuralops.core.triton_kernels.standalone.fused_act_scale_quant_gemm",
        "neuralops.core.triton_kernels.fused_hadamard_w4a16_gemm": "neuralops.core.triton_kernels.standalone.fused_hadamard_w4a16_gemm",
        "neuralops.core.triton_kernels.fused_kronecker_matmul": "neuralops.core.triton_kernels.standalone.fused_kronecker_matmul",
        "neuralops.core.triton_kernels.fused_latent_kv_cache_attention": "neuralops.core.triton_kernels.standalone.fused_latent_kv_cache_attention",
        "neuralops.core.triton_kernels.fused_lowrank_gemm": "neuralops.core.triton_kernels.standalone.fused_lowrank_gemm",
        "neuralops.core.triton_kernels.hgrn2_fused_bwd_scan": "neuralops.core.triton_kernels.standalone.hgrn2_fused_bwd_scan",
        "neuralops.core.triton_kernels.lut_codebook_dequant_gemm": "neuralops.core.triton_kernels.standalone.lut_codebook_dequant_gemm",
        "neuralops.core.triton_kernels.rpca_sparse_gemm": "neuralops.core.triton_kernels.standalone.rpca_sparse_gemm",
        "neuralops.core.triton_kernels.spectral_entropy_svd_gate": "neuralops.core.triton_kernels.standalone.spectral_entropy_svd_gate",
        "neuralops.core.triton_kernels.strided_tt": "neuralops.core.triton_kernels.standalone.strided_tt",
        "neuralops.kernels.triton.block_sparse_streaming_attn": "neuralops.core.triton_kernels.standalone.block_sparse_streaming_attn",
        "neuralops.kernels.triton.chunked_delta_gate_scan": "neuralops.core.triton_kernels.standalone.chunked_delta_gate_scan",
        "neuralops.kernels.triton.cosformer_linear_attention": "neuralops.core.triton_kernels.standalone.cosformer_linear_attention",
        "neuralops.kernels.triton.fused_act_scale_quant_gemm": "neuralops.core.triton_kernels.standalone.fused_act_scale_quant_gemm",
        "neuralops.kernels.triton.fused_hadamard_w4a16_gemm": "neuralops.core.triton_kernels.standalone.fused_hadamard_w4a16_gemm",
        "neuralops.kernels.triton.fused_kronecker_matmul": "neuralops.core.triton_kernels.standalone.fused_kronecker_matmul",
        "neuralops.kernels.triton.fused_latent_kv_cache_attention": "neuralops.core.triton_kernels.standalone.fused_latent_kv_cache_attention",
        "neuralops.kernels.triton.fused_lowrank_gemm": "neuralops.core.triton_kernels.standalone.fused_lowrank_gemm",
        "neuralops.kernels.triton.hgrn2_fused_bwd_scan": "neuralops.core.triton_kernels.standalone.hgrn2_fused_bwd_scan",
        "neuralops.kernels.triton.hybrid_local_kernel_triton": "neuralops.core.triton_kernels.standalone.hybrid_local_kernel_triton",
        "neuralops.kernels.triton.lut_codebook_dequant_gemm": "neuralops.core.triton_kernels.standalone.lut_codebook_dequant_gemm",
        "neuralops.kernels.triton.rpca_sparse_gemm": "neuralops.core.triton_kernels.standalone.rpca_sparse_gemm",
        "neuralops.kernels.triton.spectral_entropy_svd_gate": "neuralops.core.triton_kernels.standalone.spectral_entropy_svd_gate",
        "neuralops.kernels.triton.strided_tt": "neuralops.core.triton_kernels.standalone.strided_tt",
        "neuralops.kernels.triton.gla_scan": "neuralops.core.triton_kernels.gla_scan",
        "neuralops.kernels.triton.selective_scan": "neuralops.core.triton_kernels.selective_scan",
        "neuralops.kernels.triton.fms_scan": "neuralops.core.triton_kernels.fms_scan",
        "neuralops.kernels.triton.paged_attention": "neuralops.core.triton_kernels.paged_attention",
        "neuralops.kernels.triton.flash_decoding": "neuralops.core.triton_kernels.flash_decoding",
        "neuralops.kernels.triton.sliding_window": "neuralops.core.triton_kernels.sliding_window",
        "neuralops.kernels.triton.fused_ce": "neuralops.core.triton_kernels.fused_ce",
        "neuralops.kernels.triton.fused_kd": "neuralops.core.triton_kernels.fused_kd",
        "neuralops.kernels.triton.fused_norm_linear": "neuralops.core.triton_kernels.fused_norm_linear",
        "neuralops.kernels.triton.fused_rope": "neuralops.core.triton_kernels.fused_rope",
        "neuralops.kernels.triton.low_rank": "neuralops.core.triton_kernels.low_rank",
        "neuralops.kernels.triton.elementwise": "neuralops.core.triton_kernels.elementwise",
        "neuralops.kernels.triton.linear_attn": "neuralops.core.triton_kernels.linear_attn",
        "neuralops.kernels.triton.cosformer_triton_kernel": "neuralops.core.triton_kernels.cosformer_triton_kernel",
        "neuralops.kernels.triton.block_sparse": "neuralops.core.triton_kernels.block_sparse",
        "neuralops.kernels.triton.sparse_attn": "neuralops.core.triton_kernels.sparse_attn",
        "neuralops.kernels.triton.gqa": "neuralops.core.triton_kernels.gqa",
        "neuralops.kernels.triton.varlen_attention": "neuralops.core.triton_kernels.varlen_attention",
        "neuralops.kernels.triton.state_update": "neuralops.core.triton_kernels.state_update",
        "neuralops.kernels.triton.softplus": "neuralops.core.triton_kernels.softplus",
        "neuralops.kernels.triton.autotuner": "neuralops.core.triton_kernels.autotuner",
        "neuralops.kernels.triton.hardware_profile": "neuralops.core.triton_kernels.hardware_profile",
        "neuralops.kernels.triton.op_registry": "neuralops.core.triton_kernels.op_registry",
        "neuralops.kernels.triton._common": "neuralops.core.triton_kernels._common",
        "neuralops.kernels.triton._types": "neuralops.core.triton_kernels._types",
        "neuralops.kernels.triton.differential_attention": "neuralops.core.triton_kernels.differential_attention",
        "neuralops.kernels.triton.hydra_attention": "neuralops.core.triton_kernels.hydra_attention",
        "neuralops.kernels.triton.retnet_retention": "neuralops.core.triton_kernels.retnet_retention",
        "neuralops.kernels.triton.dynamic_sparse_attention": "neuralops.core.triton_kernels.dynamic_sparse_attention",
        "neuralops.kernels.triton.cur_linear": "neuralops.core.triton_kernels.cur_linear",
    }

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any | None:
        """Execute find_spec operation."""
        if fullname.startswith("neuralops.kernels.cuda."):
            sub = fullname[len("neuralops.kernels.cuda.") :]
            canonical = f"neuralops.core.cuda_kernels.{sub}"
            try:
                if not os.environ.get("NEURALOPS_NO_DEPRECATION_WARNINGS"):
                    warnings.warn(
                        f"'{fullname}' is deprecated and will be removed in v3.0.0; use '{canonical}' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                mod = importlib.import_module(canonical)
                sys.modules[fullname] = mod
                return getattr(mod, "__spec__", None)
            except (ModuleNotFoundError, ImportError) as exc:
                _logger.debug("Alias import '%s' -> '%s' failed: %s", fullname, canonical, exc)
            except Exception as exc:
                _logger.warning(
                    "Unexpected error loading canonical module '%s' for alias '%s': %s", canonical, fullname, exc
                )
                raise
        if fullname in self._ALIASES:
            canonical = self._ALIASES[fullname]
            try:
                if not os.environ.get("NEURALOPS_NO_DEPRECATION_WARNINGS"):
                    warnings.warn(
                        f"'{fullname}' is deprecated and will be removed in v3.0.0; use '{canonical}' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                mod = importlib.import_module(canonical)
                sys.modules[fullname] = mod
                return getattr(mod, "__spec__", None)
            except (ModuleNotFoundError, ImportError) as exc:
                _logger.debug("Alias import '%s' -> '%s' failed: %s", fullname, canonical, exc)
            except Exception as exc:
                _logger.warning(
                    "Unexpected error loading canonical module '%s' for alias '%s': %s", canonical, fullname, exc
                )
                raise
        return None


def install_module_alias_finder() -> None:
    if not any(isinstance(finder, _ModuleAliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ModuleAliasFinder())
