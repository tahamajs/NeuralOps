import numpy as np

from ..base import BasePruner


class MedianPruner(BasePruner):
    def __init__(self, config, n_startup_trials=5, n_warmup_steps=3) -> None:
        super().__init__(config)
        self._n_startup = n_startup_trials
        self._n_warmup = n_warmup_steps
        self._trial_results = {}

    def should_prune(self, trial_id, intermediate_result, step) -> Any:
        """Should prune."""
        self._trial_results.setdefault(trial_id, []).append(intermediate_result)
        if step < self._n_warmup:
            return False
        # Check against median of all active trials at this step
        all_values = []
        for tid, vals in self._trial_results.items():
            if len(vals) > step:
                all_values.append(vals[step])
        if len(all_values) < self._n_startup:
            return False
        median = np.median(all_values)
        return intermediate_result < median  # prune if worse than median
