import optuna
from optuna.trial import TrialState
import json

class HyperparameterOptimizer:
    def __init__(self, train_fn, eval_fn, config_template):
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.config_template = config_template

    def objective(self, trial):
        config = self.config_template.copy()
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-6, 5e-5, log=True)
        config['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32])
        config['rank'] = trial.suggest_int('rank', 32, 256, step=32)
        config['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
        config['warmup_ratio'] = trial.suggest_float('warmup_ratio', 0.0, 0.3)

        self.train_fn(config)
        metrics = self.eval_fn(config)
        return metrics.get('accuracy', metrics.get('matthews_correlation', 0.0))

    def run(self, n_trials=20, study_name='svd_attention_opt', storage=None):
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage)
        study.optimize(self.objective, n_trials=n_trials)
        with open('best_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        study.trials_dataframe().to_csv('optuna_trials.csv')
        return study.best_params