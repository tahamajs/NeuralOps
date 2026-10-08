"""Regression tests for known bugs and fixes."""

import warnings
import pytest


@pytest.mark.regression
class TestKnownBugs:
    """Regression suite covering aliases, refactored split exports, and lazy loading."""

    def test_aegis_v2_alias_works(self):
        """Regression: aegis_v2 should resolve to hybrid_local_kernel."""
        from neuralops.core.registry import ATTENTION_REGISTRY
        import neuralops.attentions  # noqa: F401

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls_old = ATTENTION_REGISTRY.get("aegis_v2")
            cls_new = ATTENTION_REGISTRY.get("hybrid_local_kernel")
        assert cls_old is cls_new

    def test_leka_aliases_work(self):
        """Regression: leka_v2/v3 should resolve."""
        from neuralops.core.registry import ATTENTION_REGISTRY
        import neuralops.attentions  # noqa: F401

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert ATTENTION_REGISTRY.get("leka_v2") is not None
            assert ATTENTION_REGISTRY.get("leka_v3") is not None

    def test_eval_file_split_preserves_api(self):
        """Regression: eval file split must preserve public API."""
        from neuralops.evaluation.eval_all_models_datasets import (
            TRACK_MODULES,
            apply_module_to_model,
            evaluate_single_combination_opt,
        )

        assert callable(apply_module_to_model)
        assert callable(evaluate_single_combination_opt)
        assert TRACK_MODULES is not None

    def test_legacy_runner_shims_work(self):
        """Regression: legacy runners should still import."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from neuralops.benchmarks import run_gpt2_sweep  # noqa: F401
            from neuralops.benchmarks import run_quality_sweep  # noqa: F401

    def test_registry_lazy_load_works(self):
        """Regression: registry should lazy-load without circular import."""
        from neuralops.core.registry import ATTENTION_REGISTRY

        assert len(ATTENTION_REGISTRY.keys()) > 0
