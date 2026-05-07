import optuna
import os
from django.conf import settings
from experiment.models import Sweep

def get_study(sweep):
    """Return or create optuna Study for the given sweep."""
    storage_path = os.path.join(
        settings.MEDIA_ROOT, 'optuna', f'sweep_{sweep.id}.db'
    )
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    storage_url = f'sqlite:///{storage_path}'

    study = optuna.create_study(
        study_name=f'sweep_{sweep.id}',
        storage=storage_url,
        direction=sweep.direction,
        load_if_exists=True,
    )
    return study


def suggest_params(sweep):
    """Create a new optuna trial and return a parameter dict."""
    study = get_study(sweep)
    trial = study.ask()
    # Convert trial.params to a plain dict
    params = {}
    for key, value in trial.params.items():
        params[key] = value
    # Attach trial number to the key for later reference
    params['_trial_number'] = trial.number
    return params, trial


def report_metric(sweep, trial_number, value):
    """Report a completed trial's objective value back to Optuna."""
    study = get_study(sweep)
    # Recreate the trial object using its number
    trials = study.get_trials(deepcopy=False)
    for t in trials:
        if t.number == trial_number:
            study.tell(t, value)
            return
    # If not found, create a new trial and report
    trial = study.ask()
    study.tell(trial, value)


def get_parameter_importance(sweep):
    """Return hyperparameter importance dict from Optuna."""
    study = get_study(sweep)
    try:
        return optuna.importance.get_param_importances(study)
    except Exception:
        return {}