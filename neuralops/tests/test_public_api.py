"""Smoke tests for newly-exported public API functions."""
import pytest


class TestTrainerFactories:
    def test_optimizer_factories_importable(self):
        from neuralops.trainer import (
            create_adabelief,
            create_adam,
            create_adamw,
            create_lamb,
            create_lars,
            create_muon,
            create_muon_adamw_reg,
            create_ranger,
            create_riemannian_stiefel,
            create_sgd,
        )
        assert all(
            callable(f)
            for f in [
                create_muon,
                create_muon_adamw_reg,
                create_riemannian_stiefel,
                create_adamw,
                create_sgd,
                create_adam,
                create_adabelief,
                create_ranger,
                create_lamb,
                create_lars,
            ]
        )

    def test_scheduler_factories_importable(self):
        from neuralops.trainer import (
            create_cosine,
            create_linear,
            create_onecycle,
            create_poly,
            create_rop,
        )
        assert all(
            callable(f)
            for f in [
                create_linear,
                create_cosine,
                create_rop,
                create_onecycle,
                create_poly,
            ]
        )


class TestStatisticalUtilities:
    def test_corrections_importable(self):
        from neuralops.analysis.comparison import (
            finner_correction,
            hochberg_correction,
            holland_copenhaver_correction,
            holm_correction,
            hommel_correction,
        )
        assert all(
            callable(f)
            for f in [
                holm_correction,
                hochberg_correction,
                finner_correction,
                hommel_correction,
                holland_copenhaver_correction,
            ]
        )

    def test_hypothesis_tests_importable(self):
        from neuralops.analysis.comparison import (
            bayesian_correlated_ttest,
            bayesian_signed_rank_test,
            mcnemar_test,
            run_full_statistical_analysis,
            sign_test,
            wilcoxon_signed_rank_test,
        )
        assert all(
            callable(f)
            for f in [
                wilcoxon_signed_rank_test,
                sign_test,
                mcnemar_test,
                bayesian_correlated_ttest,
                bayesian_signed_rank_test,
                run_full_statistical_analysis,
            ]
        )


class TestUtilityHelpers:
    def test_io_helpers(self):
        from neuralops.utils import format_time, load_json, save_json
        assert all(callable(f) for f in [save_json, load_json, format_time])

    def test_profiling_helpers(self):
        from neuralops.utils import measure_memory_usage, profile_context
        assert callable(measure_memory_usage) and callable(profile_context)

    def test_architecture_helpers(self):
        from neuralops.utils import get_compression_targets, get_hidden_size, get_layer_count, get_model_info
        assert all(
            callable(f)
            for f in [get_model_info, get_layer_count, get_hidden_size, get_compression_targets]
        )

    def test_qwen_helpers(self):
        from neuralops.utils import is_architecture_error, is_qwen_attention, qwen_config_proxy
        assert all(callable(f) for f in [is_qwen_attention, qwen_config_proxy, is_architecture_error])

    def test_data_helpers(self):
        from neuralops.data import (
            compute_empirical_entropy,
            generate_zipfian_markov_tokens,
            list_local_datasets,
            load_glue_split,
            load_superglue_dataset,
        )
        assert all(
            callable(f)
            for f in [
                list_local_datasets,
                load_glue_split,
                load_superglue_dataset,
                generate_zipfian_markov_tokens,
                compute_empirical_entropy,
            ]
        )
