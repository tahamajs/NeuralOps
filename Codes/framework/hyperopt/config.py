from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

class HPOBackend(Enum):
    OPTUNA = "optuna"
    RAY_TUNE = "ray_tune"
    HYPEROPT = "hyperopt"
    SCIKIT_OPTIMIZE = "scikit_optimize"
    NEVERGRAD = "nevergrad"

class HPOAlgorithm(Enum):
    TPE = "tpe"                    # Tree‑structured Parzen Estimator
    RANDOM = "random"
    BAYESIAN = "bayesian"          # Gaussian Process
    CMA_ES = "cma_es"              # Covariance Matrix Adaptation
    HYPERBAND = "hyperband"
    ASHA = "asha"                  # Asynchronous Successive Halving
    BOHB = "bohb"                  # Bayesian Optimization + HyperBand
    NGOpt = "ngopt"                # Nevergrad's adaptive algorithm
    PSO = "pso"
    DE = "de"                      # Differential Evolution

class PruningMethod(Enum):
    NONE = "none"
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    THRESHOLD = "threshold"

@dataclass
class HPOConfig:
    """Complete configuration for hyperparameter optimization."""
    backend: HPOBackend = HPOBackend.OPTUNA
    algorithm: HPOAlgorithm = HPOAlgorithm.TPE
    num_trials: int = 50
    search_space: Dict[str, Any] = field(default_factory=dict)
    direction: str = "maximize"    # "maximize" or "minimize"
    pruning: PruningMethod = PruningMethod.NONE
    pruner_n_startup_trials: int = 10
    pruner_n_warmup_steps: int = 5
    pruner_interval_steps: int = 1
    multi_objective: bool = False
    objectives: List[str] = field(default_factory=list)
    num_parallel: int = 1
    cv_folds: int = 0
    seed: int = 42
    timeout_seconds: Optional[int] = None
    storage_path: Optional[str] = None
    report_to_saver: bool = True