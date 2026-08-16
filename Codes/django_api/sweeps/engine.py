import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False

from django.conf import settings
from .models import Sweep

logger = logging.getLogger(__name__)


# ── Storage ─────────────────────────────────────────────────────────
def get_storage_path(sweep_id) -> str:
    base = os.path.join(settings.MEDIA_ROOT, "optuna")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"sweep_{sweep_id}.db")


# ── Sampler Factory ──────────────────────────────────────────────────
if HAS_OPTUNA:
    SAMPLER_MAP = {
        "tpe": optuna.samplers.TPESampler,
        "cmaes": optuna.samplers.CmaEsSampler,
        "nsgaii": optuna.samplers.NSGAIISampler,
        "nsgaiii": optuna.samplers.NSGAIIISampler,
        "grid": optuna.samplers.GridSampler,
        "random": optuna.samplers.RandomSampler,
        "qmc": optuna.samplers.QMCSampler,
        "gp": optuna.samplers.GPSampler,
        "botorch": getattr(optuna.integration, "BoTorchSampler", optuna.samplers.TPESampler) if hasattr(optuna, "integration") else optuna.samplers.TPESampler,
        "auto": None,  # handled separately
    }

    PRUNER_MAP = {
        "hyperband": optuna.pruners.HyperbandPruner,
        "median": optuna.pruners.MedianPruner,
        "wilcoxon": getattr(optuna.pruners, "WilcoxonPruner", optuna.pruners.MedianPruner),
        "successive_halving": optuna.pruners.SuccessiveHalvingPruner,
        "none": optuna.pruners.NopPruner,
    }
else:
    SAMPLER_MAP = {}
    PRUNER_MAP = {}


def _build_sampler(sweep: Sweep):
    sampler_cls = SAMPLER_MAP.get(sweep.algorithm)
    if sampler_cls is None and sweep.algorithm == "auto":
        # AutoSampler picks the best sampler automatically
        return optuna.samplers.AutoSampler(seed=42)
    if sampler_cls is None:
        logger.warning("Unknown sampler '%s', falling back to TPE", sweep.algorithm)
        sampler_cls = optuna.samplers.TPESampler
    kwargs = {"seed": sweep.config.get("sampler_seed", 42)}
    # GPSampler supports constraints
    if sweep.algorithm == "gp":
        constraints = sweep.config.get("constraints", [])
        if constraints:
            kwargs["constraints_func"] = lambda trial: _eval_constraints(
                trial, constraints
            )
    return sampler_cls(**kwargs)


def _build_pruner(sweep: Sweep):
    pruner_cls = PRUNER_MAP.get(sweep.pruner, optuna.pruners.NopPruner)
    kwargs = {}
    if sweep.pruner == "hyperband":
        kwargs["min_resource"] = sweep.config.get("min_resource", 1)
        kwargs["max_resource"] = sweep.config.get("max_resource", "auto")
        kwargs["reduction_factor"] = sweep.config.get("reduction_factor", 3)
    elif sweep.pruner == "median":
        kwargs["n_startup_trials"] = sweep.config.get("n_startup_trials", 5)
        kwargs["n_warmup_steps"] = sweep.config.get("n_warmup_steps", 0)
    return pruner_cls(**kwargs)


def _eval_constraints(trial, constraints: List[Dict]) -> List[float]:
    """Evaluate constraint functions. Returns list of values where ≤0 indicates feasible."""
    results = []
    for c in constraints:
        param_name = c["param"]
        threshold = c.get("threshold", 0)
        op = c.get("op", "le")
        val = trial.params.get(param_name, 0)
        if op == "le":
            results.append(val - threshold)
        elif op == "ge":
            results.append(threshold - val)
        else:
            results.append(0.0)
    return results


# ── Study Creation ───────────────────────────────────────────────────
def get_study(sweep: Sweep) -> optuna.Study:
    storage = f"sqlite:///{get_storage_path(sweep.id)}"
    sampler = _build_sampler(sweep)
    pruner = _build_pruner(sweep)

    directions = []
    if sweep.metric_names:
        for d in sweep.directions:
            directions.append(d)
    else:
        directions = [sweep.direction]

    study = optuna.create_study(
        study_name=f"sweep_{sweep.id}",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        directions=directions,
        load_if_exists=True,
    )
    return study


# ── Trial Suggestion ─────────────────────────────────────────────────
def suggest_params(sweep: Sweep) -> Tuple[Dict[str, Any], optuna.trial.Trial]:
    study = get_study(sweep)
    search_space = sweep.config.get("search_space", {})

    def _suggest(trial):
        params = {}
        for name, spec in search_space.items():
            ptype = spec.get("type", "float")
            if "condition" in spec:
                cond_param = spec["condition"].get("param")
                cond_value = spec["condition"].get("value")
                if cond_param in params and params[cond_param] != cond_value:
                    continue
            if ptype == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec.get("low", 0),
                    spec.get("high", 1),
                    log=spec.get("log", False),
                    step=spec.get("step", None),
                )
            elif ptype == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec.get("low", 0),
                    spec.get("high", 100),
                    step=spec.get("step", 1),
                    log=spec.get("log", False),
                )
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, spec.get("choices", []))
            elif ptype == "discrete_uniform":
                params[name] = trial.suggest_discrete_uniform(
                    name, spec.get("low", 0), spec.get("high", 1), spec.get("q", 0.1)
                )
            elif ptype == "uniform":
                params[name] = trial.suggest_float(
                    name, spec.get("low", 0), spec.get("high", 1)
                )
        return params

    trial = study.ask()
    params = _suggest(trial)
    return params, trial


# ── Metric Reporting ─────────────────────────────────────────────────
def report_metric(
    sweep: Sweep,
    trial_number: int,
    objective_value: Optional[float] = None,
    objectives: Optional[List[float]] = None,
    user_attrs: Optional[Dict] = None,
):
    study = get_study(sweep)
    frozen_trial = None
    for t in study.get_trials(states=None):
        if t.number == trial_number:
            frozen_trial = t
            break
    if frozen_trial is None:
        logger.warning("Trial %d not found in study", trial_number)
        return
    if objectives is not None:
        study.tell(frozen_trial, objectives)
    else:
        study.tell(frozen_trial, objective_value)
    if user_attrs:
        for k, v in user_attrs.items():
            frozen_trial.set_user_attr(k, v)


# ── Intermediate Reporting ──────────────────────────────────────────
def report_intermediate(sweep: Sweep, trial_number: int, step: int, value: float):
    study = get_study(sweep)
    for t in study.get_trials(states=None):
        if t.number == trial_number:
            t.report(value, step)
            return


# ── Trial Pruning ────────────────────────────────────────────────────
def should_prune(sweep: Sweep, trial_number: int) -> bool:
    study = get_study(sweep)
    for t in study.get_trials(states=None):
        if t.number == trial_number:
            return t.should_prune()
    return False


# ── Visualization Helpers ────────────────────────────────────────────
def get_parameter_importance(sweep: Sweep) -> Dict:
    study = get_study(sweep)
    try:
        return optuna.importance.get_param_importances(study)
    except Exception:
        return {}


def get_pareto_front(sweep: Sweep) -> List[Dict]:
    if not sweep.metric_names:
        return []
    study = get_study(sweep)
    try:
        trials = [t for t in study.trials if t.values is not None]
        pareto = optuna.visualization._pareto_front.get_pareto_front_trials(trials)
        result = []
        for t in pareto:
            result.append(
                {
                    "number": t.number,
                    "params": t.params,
                    "objectives": t.values,
                }
            )
        return result
    except Exception:
        return []


def get_trial_contour(sweep: Sweep, params: List[str]) -> Dict:
    study = get_study(sweep)
    try:
        fig = optuna.visualization.plot_contour(study, params=params)
        return json.loads(fig.to_json())
    except Exception:
        return {}


def get_optimization_history(sweep: Sweep) -> Dict:
    study = get_study(sweep)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        return json.loads(fig.to_json())
    except Exception:
        return {}


def get_slice_plot(sweep: Sweep, param: str) -> Dict:
    study = get_study(sweep)
    try:
        fig = optuna.visualization.plot_slice(study, params=[param])
        return json.loads(fig.to_json())
    except Exception:
        return {}


def get_parallel_coordinate(sweep: Sweep) -> Dict:
    study = get_study(sweep)
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        return json.loads(fig.to_json())
    except Exception:
        return {}


def get_empirical_attainment_surface(sweep: Sweep) -> Dict:
    """For multi-objective sweeps only."""
    study = get_study(sweep)
    try:
        from optuna.visualization import plot_edf

        fig = plot_edf(study)
        return json.loads(fig.to_json())
    except Exception:
        return {}


def get_warm_start_config(sweep: Sweep) -> Dict:
    study = get_study(sweep)
    best = study.best_trial
    if best:
        return {
            "best_params": best.params,
            "best_value": best.value,
            "trial_number": best.number,
        }
    return {}


def get_study_dataframe(sweep: Sweep) -> List[Dict]:
    study = get_study(sweep)
    df = study.trials_dataframe()
    return df.to_dict(orient="records")


def export_study(sweep: Sweep, output_path: str):
    """Export study to a JSON file."""
    study = get_study(sweep)
    data = {
        "name": sweep.name,
        "direction": sweep.direction,
        "directions": sweep.directions,
        "metric_name": sweep.metric_name,
        "metric_names": sweep.metric_names,
        "algorithm": sweep.algorithm,
        "pruner": sweep.pruner,
        "trials": [],
    }
    for t in study.trials:
        data["trials"].append(
            {
                "number": t.number,
                "params": t.params,
                "values": t.values,
                "state": str(t.state),
                "user_attrs": t.user_attrs,
            }
        )
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def import_study(sweep: Sweep, input_path: str):
    """Import trials from a JSON file into the study."""
    with open(input_path) as f:
        data = json.load(f)
    study = get_study(sweep)
    for t_data in data.get("trials", []):
        trial = study.ask()
        trial.suggest_float("_dummy", 0, 1)
        study.tell(trial, t_data["values"][0] if t_data.get("values") else 0)
